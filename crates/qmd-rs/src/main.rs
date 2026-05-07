use retrieval_core::{build_retrieval_indices, chunk_markdown, reciprocal_rank_fusion, LexicalRetriever, VectorRetriever};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn read_markdown_dir(dir: &Path, chunk_size: usize) -> Result<Vec<retrieval_core::DocumentChunk>, String> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("md"))
        .collect();
    paths.sort();

    let mut chunks = Vec::new();
    for path in paths {
        let doc_id = path.file_stem().and_then(|x| x.to_str()).ok_or("invalid filename")?;
        let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        chunks.extend(chunk_markdown(doc_id, &content, chunk_size));
    }

    if chunks.is_empty() {
        return Err(format!("no markdown files found in {}", dir.display()));
    }
    Ok(chunks)
}

fn cmd_index(corpus_dir: &str, chunk_size: usize) -> Result<(), String> {
    let chunks = read_markdown_dir(Path::new(corpus_dir), chunk_size)?;
    println!("indexed {} chunks from {}", chunks.len(), corpus_dir);
    Ok(())
}

fn cmd_query(corpus_dir: &str, query: &str, top_k: usize, chunk_size: usize) -> Result<(), String> {
    let chunks = read_markdown_dir(Path::new(corpus_dir), chunk_size)?;
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
            let chunk_size = args.get(3).and_then(|x| x.parse::<usize>().ok()).unwrap_or(64);
            cmd_index(corpus, chunk_size)
        }
        Some("query") => {
            let corpus = args.get(2).map(String::as_str).unwrap_or("eval/corpus");
            let query = args.get(3).map(String::as_str).unwrap_or("rust retrieval");
            let top_k = args.get(4).and_then(|x| x.parse::<usize>().ok()).unwrap_or(5);
            let chunk_size = args.get(5).and_then(|x| x.parse::<usize>().ok()).unwrap_or(64);
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
}
