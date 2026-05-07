use retrieval_core::{
    build_retrieval_indices, chunk_record, parse_markdown_record, reciprocal_rank_fusion,
    LexicalRetriever, VectorRetriever,
};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, PartialEq, Eq)]
struct IndexMetadata {
    chunk_size: usize,
    corpus_fingerprint: String,
}

#[derive(Debug, PartialEq, Eq)]
struct PluginRequest {
    query: String,
    top_k: usize,
    corpus_dir: String,
}

fn log_event(level: &str, msg: &str) {
    eprintln!("{{\"level\":\"{}\",\"msg\":\"{}\"}}", level, msg);
}

fn parse_plugin_request(body: &str) -> Result<PluginRequest, String> {
    let query = extract_json_string(body, "query").ok_or("missing query")?;
    let top_k = extract_json_usize(body, "top_k").unwrap_or(5);
    let corpus_dir = extract_json_string(body, "corpus_dir").unwrap_or_else(|| "eval/corpus".into());
    Ok(PluginRequest {
        query,
        top_k,
        corpus_dir,
    })
}

fn extract_json_string(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let i = body.find(&needle)?;
    let rest = &body[i + needle.len()..];
    let q = rest.find('"')?;
    let rest = &rest[q + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_json_usize(body: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let i = body.find(&needle)?;
    let rest = &body[i + needle.len()..];
    let c = rest.find(':')?;
    let rest = rest[c + 1..].trim_start();
    let n: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    n.parse().ok()
}

fn markdown_paths(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("md"))
        .collect();
    paths.sort();
    Ok(paths)
}

fn corpus_fingerprint(paths: &[PathBuf]) -> Result<String, String> {
    let mut acc = String::new();
    for path in paths {
        let meta = fs::metadata(path).map_err(|e| e.to_string())?;
        let modified = meta
            .modified()
            .map_err(|e| e.to_string())?
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();
        acc.push_str(&format!("{}:{}:{}|", path.display(), meta.len(), modified));
    }
    Ok(acc)
}

fn read_markdown_dir(
    dir: &Path,
    chunk_size: usize,
) -> Result<(Vec<retrieval_core::DocumentChunk>, IndexMetadata), String> {
    let paths = markdown_paths(dir)?;
    let mut chunks = Vec::new();
    for path in &paths {
        let doc_id = path
            .file_stem()
            .and_then(|x| x.to_str())
            .ok_or("invalid filename")?;
        let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let record = parse_markdown_record(doc_id, &content);
        chunks.extend(chunk_record(&record, chunk_size));
    }

    if chunks.is_empty() {
        return Err(format!("no markdown files found in {}", dir.display()));
    }

    Ok((
        chunks,
        IndexMetadata {
            chunk_size,
            corpus_fingerprint: corpus_fingerprint(&paths)?,
        },
    ))
}

fn write_chunk_index(
    output_path: &Path,
    metadata: &IndexMetadata,
    chunks: &[retrieval_core::DocumentChunk],
) -> Result<(), String> {
    let mut out = format!(
        "#meta\tchunk_size={}\tfingerprint={}\n",
        metadata.chunk_size,
        escape_index_field(&metadata.corpus_fingerprint)
    );
    for c in chunks {
        out.push_str(&c.id);
        out.push('\t');
        out.push_str(&escape_index_field(&c.text));
        out.push('\n');
    }
    fs::write(output_path, out).map_err(|e| e.to_string())
}

fn read_chunk_index(
    index_path: &Path,
) -> Result<(IndexMetadata, Vec<retrieval_core::DocumentChunk>), String> {
    let raw = fs::read_to_string(index_path).map_err(|e| e.to_string())?;
    let mut lines = raw.lines();
    let metadata = parse_index_metadata(lines.next().ok_or("missing metadata header in index")?)?;

    let mut chunks = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        chunks.push(retrieval_core::DocumentChunk {
            id: parts.next().ok_or("missing chunk id")?.to_string(),
            text: unescape_index_field(parts.next().ok_or("missing chunk text")?)?,
        });
    }
    if chunks.is_empty() {
        return Err(format!("index {} is empty", index_path.display()));
    }
    Ok((metadata, chunks))
}

fn parse_index_metadata(meta_line: &str) -> Result<IndexMetadata, String> {
    let mut parts = meta_line.split('\t');
    if parts.next() != Some("#meta") {
        return Err("invalid metadata prefix".into());
    }
    let cs = parts
        .next()
        .and_then(|s| s.strip_prefix("chunk_size="))
        .ok_or("missing chunk_size in metadata")?;
    let fp = parts
        .next()
        .and_then(|s| s.strip_prefix("fingerprint="))
        .ok_or("missing fingerprint in metadata")?;

    Ok(IndexMetadata {
        chunk_size: cs.parse().map_err(|_| "invalid chunk_size in metadata")?,
        corpus_fingerprint: unescape_index_field(fp)?,
    })
}

fn escape_index_field(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn unescape_index_field(input: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some(other) => return Err(format!("invalid escape sequence: \\{other}")),
            None => return Err("dangling escape sequence in index".into()),
        }
    }
    Ok(out)
}

fn startup_self_checks() -> Result<(), String> {
    inference_runtime::ModelCache::new(".rqmd-models").ensure_layout()?;
    log_event("info", "startup_checks_ok");
    Ok(())
}

fn retrieve(
    corpus_dir: &str,
    query: &str,
    top_k: usize,
    chunk_size: usize,
) -> Result<Vec<(String, f32)>, String> {
    let index_path = Path::new(corpus_dir).join(".rqmd_chunks.tsv");
    let (chunks, refresh) = if index_path.exists() {
        let (m, c) = read_chunk_index(&index_path)?;
        let paths = markdown_paths(Path::new(corpus_dir))?;
        let cur = IndexMetadata {
            chunk_size,
            corpus_fingerprint: corpus_fingerprint(&paths)?,
        };
        (c, m != cur)
    } else {
        (Vec::new(), true)
    };

    let chunks = if refresh {
        let (fresh, m) = read_markdown_dir(Path::new(corpus_dir), chunk_size)?;
        write_chunk_index(&index_path, &m, &fresh)?;
        fresh
    } else {
        chunks
    };

    let (bm25, ann) = build_retrieval_indices(chunks);
    let fused = reciprocal_rank_fusion(
        &bm25.search(query, top_k.max(1)),
        &ann.search(query, top_k.max(1)),
        60.0,
    );
    Ok(fused
        .into_iter()
        .take(top_k.max(1))
        .map(|r| (r.id, r.score))
        .collect())
}

fn cmd_index(corpus_dir: &str, chunk_size: usize) -> Result<(), String> {
    let (chunks, metadata) = read_markdown_dir(Path::new(corpus_dir), chunk_size)?;
    let index_path = Path::new(corpus_dir).join(".rqmd_chunks.tsv");
    write_chunk_index(&index_path, &metadata, &chunks)?;
    println!(
        "indexed {} chunks from {} -> {}",
        chunks.len(),
        corpus_dir,
        index_path.display()
    );
    Ok(())
}

fn cmd_query(corpus_dir: &str, query: &str, top_k: usize, chunk_size: usize) -> Result<(), String> {
    let rows = retrieve(corpus_dir, query, top_k, chunk_size)?;
    if rows.is_empty() {
        println!("no matches");
        return Ok(());
    }
    for (id, score) in rows {
        println!("{}\t{:.4}", id, score);
    }
    Ok(())
}

fn cmd_plugin() -> Result<(), String> {
    let mut body = String::new();
    std::io::stdin()
        .read_to_string(&mut body)
        .map_err(|e| e.to_string())?;
    let request = parse_plugin_request(&body)?;

    if env::var("RQMD_FORCE_TS_FALLBACK").ok().as_deref() == Some("1") {
        println!("{{\"results\":[],\"fallback\":\"ts:forced\"}}");
        return Ok(());
    }

    match retrieve(&request.corpus_dir, &request.query, request.top_k, 64) {
        Ok(results) => {
            let items = results
                .into_iter()
                .map(|(id, score)| format!("{{\"id\":\"{}\",\"score\":{}}}", id, score))
                .collect::<Vec<_>>()
                .join(",");
            println!("{{\"results\":[{}]}}", items);
        }
        Err(err) => {
            log_event("warn", &format!("plugin_retrieve_failed:{err}"));
            println!("{{\"results\":[],\"fallback\":\"ts:auto\"}}");
        }
    }
    Ok(())
}

fn handle_http(mut stream: TcpStream) -> Result<(), String> {
    let mut buf = [0_u8; 1024];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let (st, body) = if req.starts_with("GET /health") {
        ("200 OK", "{\"status\":\"ok\"}")
    } else {
        ("404 Not Found", "{\"status\":\"not_found\"}")
    };
    let rsp = format!(
        "HTTP/1.1 {st}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(rsp.as_bytes()).map_err(|e| e.to_string())
}

fn cmd_serve(bind: &str) -> Result<(), String> {
    let listener = TcpListener::bind(bind).map_err(|e| e.to_string())?;
    println!("rpc listening on {bind}");
    for conn in listener.incoming() {
        handle_http(conn.map_err(|e| e.to_string())?)?;
    }
    Ok(())
}

fn main() {
    let _ = startup_self_checks();

    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("index") => cmd_index(
            args.get(2).map(String::as_str).unwrap_or("eval/corpus"),
            args.get(3).and_then(|x| x.parse().ok()).unwrap_or(64),
        ),
        Some("query") => cmd_query(
            args.get(2).map(String::as_str).unwrap_or("eval/corpus"),
            args.get(3).map(String::as_str).unwrap_or("rust retrieval"),
            args.get(4).and_then(|x| x.parse().ok()).unwrap_or(5),
            args.get(5).and_then(|x| x.parse().ok()).unwrap_or(64),
        ),
        Some("serve") => cmd_serve(args.get(2).map(String::as_str).unwrap_or("127.0.0.1:8080")),
        Some("plugin") => cmd_plugin(),
        _ => {
            eprintln!("usage: rqmd <index|query|serve|plugin>");
            Err("invalid command".into())
        }
    };

    if let Err(err) = result {
        eprintln!("error: {err}");
        std::process::exit(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plugin_request_defaults() {
        let req = parse_plugin_request("{\"query\":\"rust\"}").expect("parse");
        assert_eq!(req.query, "rust");
        assert_eq!(req.top_k, 5);
        assert_eq!(req.corpus_dir, "eval/corpus");
    }
}
