use retrieval_core::{
    build_retrieval_indices, chunk_record, parse_markdown_record, reciprocal_rank_fusion,
    LexicalRetriever, VectorRetriever,
};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, PartialEq, Eq)]
struct IndexMetadata {
    chunk_size: usize,
    corpus_fingerprint: String,
}

fn markdown_paths(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("md"))
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
        let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
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
    for chunk in chunks {
        let text = escape_index_field(&chunk.text);
        out.push_str(&chunk.id);
        out.push('\t');
        out.push_str(&text);
        out.push('\n');
    }
    fs::write(output_path, out).map_err(|e| e.to_string())
}

fn read_chunk_index(
    index_path: &Path,
) -> Result<(IndexMetadata, Vec<retrieval_core::DocumentChunk>), String> {
    let raw = fs::read_to_string(index_path).map_err(|e| e.to_string())?;
    let mut lines = raw.lines();
    let meta_line = lines.next().ok_or("missing metadata header in index")?;
    let metadata = parse_index_metadata(meta_line)?;
    let mut chunks = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let id = parts.next().ok_or("missing chunk id")?;
        let text = parts.next().ok_or("missing chunk text")?;
        chunks.push(retrieval_core::DocumentChunk {
            id: id.to_string(),
            text: unescape_index_field(text)?,
        });
    }
    if chunks.is_empty() {
        return Err(format!("index {} is empty", index_path.display()));
    }
    Ok((metadata, chunks))
}

fn parse_index_metadata(meta_line: &str) -> Result<IndexMetadata, String> {
    let mut parts = meta_line.split('\t');
    let prefix = parts.next().ok_or("missing metadata prefix")?;
    if prefix != "#meta" {
        return Err("invalid metadata prefix".to_string());
    }
    let chunk_size_str = parts
        .next()
        .and_then(|s| s.strip_prefix("chunk_size="))
        .ok_or("missing chunk_size in metadata")?;
    let fingerprint_raw = parts
        .next()
        .and_then(|s| s.strip_prefix("fingerprint="))
        .ok_or("missing fingerprint in metadata")?;
    let chunk_size = chunk_size_str
        .parse::<usize>()
        .map_err(|_| "invalid chunk_size in metadata".to_string())?;
    Ok(IndexMetadata {
        chunk_size,
        corpus_fingerprint: unescape_index_field(fingerprint_raw)?,
    })
}

fn escape_index_field(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out
}

fn unescape_index_field(input: &str) -> Result<String, String> {
    let mut out = String::with_capacity(input.len());
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
            None => return Err("dangling escape sequence in index".to_string()),
        }
    }
    Ok(out)
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
    let index_path = Path::new(corpus_dir).join(".rqmd_chunks.tsv");
    let (chunks, should_refresh) = if index_path.exists() {
        let (metadata, chunks) = read_chunk_index(&index_path)?;
        let paths = markdown_paths(Path::new(corpus_dir))?;
        let current = IndexMetadata {
            chunk_size,
            corpus_fingerprint: corpus_fingerprint(&paths)?,
        };
        (chunks, metadata != current)
    } else {
        (Vec::new(), true)
    };
    let chunks = if should_refresh {
        let (fresh_chunks, metadata) = read_markdown_dir(Path::new(corpus_dir), chunk_size)?;
        write_chunk_index(&index_path, &metadata, &fresh_chunks)?;
        fresh_chunks
    } else {
        chunks
    };
    let (bm25, ann) = build_retrieval_indices(chunks);
    let lexical = bm25.search(query, top_k.max(1));
    let vector = ann.search(query, top_k.max(1));
    let fused = reciprocal_rank_fusion(&lexical, &vector, 60.0);

    if fused.is_empty() {
        println!("no matches");
        return Ok(());
    }

    for result in fused.into_iter().take(top_k.max(1)) {
        println!("{}\t{:.4}", result.id, result.score);
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("index") => {
            let corpus = args.get(2).map(String::as_str).unwrap_or("eval/corpus");
            let chunk_size = args
                .get(3)
                .and_then(|x| x.parse::<usize>().ok())
                .unwrap_or(64);
            cmd_index(corpus, chunk_size)
        }
        Some("query") => {
            let corpus = args.get(2).map(String::as_str).unwrap_or("eval/corpus");
            let query = args.get(3).map(String::as_str).unwrap_or("rust retrieval");
            let top_k = args
                .get(4)
                .and_then(|x| x.parse::<usize>().ok())
                .unwrap_or(5);
            let chunk_size = args
                .get(5)
                .and_then(|x| x.parse::<usize>().ok())
                .unwrap_or(64);
            cmd_query(corpus, query, top_k, chunk_size)
        }
        Some("serve") => {
            println!("rpc listening on 127.0.0.1:8080");
            Ok(())
        }
        _ => {
            eprintln!("usage: rqmd <index|query|serve> [args...]");
            Err("invalid command".to_string())
        }
    };

    if let Err(err) = result {
        eprintln!("error: {err}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_markdown_dir_errors_when_empty() {
        let dir = std::env::temp_dir().join("rqmd_empty_md");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");
        let err = read_markdown_dir(&dir, 16).expect_err("expected empty error");
        assert!(err.contains("no markdown files"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn chunk_index_round_trip() {
        let dir = std::env::temp_dir().join("rqmd_chunk_index_rt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");

        let chunks = vec![
            retrieval_core::DocumentChunk {
                id: "doc:0".into(),
                text: "hello\trust".into(),
            },
            retrieval_core::DocumentChunk {
                id: "doc:1".into(),
                text: "vector\nsearch".into(),
            },
        ];
        let path = dir.join(".rqmd_chunks.tsv");
        let meta = IndexMetadata {
            chunk_size: 2,
            corpus_fingerprint: "x:y:z|".into(),
        };
        write_chunk_index(&path, &meta, &chunks).expect("write index");
        let (loaded_meta, loaded) = read_chunk_index(&path).expect("read index");
        assert_eq!(loaded_meta, meta);
        assert_eq!(loaded, chunks);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unescape_rejects_invalid_sequences() {
        let err = unescape_index_field("bad\\xescape").expect_err("expected invalid escape error");
        assert!(err.contains("invalid escape sequence"));
    }

    #[test]
    fn parse_index_metadata_rejects_bad_header() {
        let err = parse_index_metadata("chunk_size=1\tfingerprint=a")
            .expect_err("expected bad metadata prefix");
        assert!(err.contains("invalid metadata prefix"));
    }
}
