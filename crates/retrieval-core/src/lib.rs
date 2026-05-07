use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentChunk {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RetrievalResult {
    pub id: String,
    pub score: f32,
}

pub fn chunk_markdown(doc_id: &str, content: &str, chunk_size: usize) -> Vec<DocumentChunk> {
    let words: Vec<&str> = content.split_whitespace().collect();
    words
        .chunks(chunk_size.max(1))
        .enumerate()
        .map(|(i, chunk)| DocumentChunk { id: format!("{doc_id}:{i}"), text: chunk.join(" ") })
        .collect()
}

pub fn normalize_tokens(input: &str) -> Vec<String> {
    input
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

pub trait LexicalRetriever: Send + Sync { fn search(&self, query: &str, top_k: usize) -> Vec<RetrievalResult>; }
pub trait VectorRetriever: Send + Sync { fn search(&self, embedding: &[f32], top_k: usize) -> Vec<RetrievalResult>; }

pub fn lexical_score(query: &str, chunk: &DocumentChunk) -> f32 {
    let q = normalize_tokens(query);
    let d = normalize_tokens(&chunk.text);
    q.iter().filter(|t| d.contains(t)).count() as f32
}

pub fn weighted_fusion(lexical: &[RetrievalResult], vector: &[RetrievalResult], lw: f32, vw: f32) -> Vec<RetrievalResult> {
    let mut acc: HashMap<&str, f32> = HashMap::new();
    for r in lexical { *acc.entry(&r.id).or_default() += r.score * lw; }
    for r in vector { *acc.entry(&r.id).or_default() += r.score * vw; }
    let mut out: Vec<RetrievalResult> = acc.into_iter().map(|(id, score)| RetrievalResult { id: id.into(), score }).collect();
    out.sort_by(|a,b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    out
}

pub fn reciprocal_rank_fusion(lexical: &[RetrievalResult], vector: &[RetrievalResult], k: f32) -> Vec<RetrievalResult> {
    let l: Vec<_> = lexical.iter().enumerate().map(|(i,r)| RetrievalResult{id:r.id.clone(),score:1.0/(k+i as f32+1.0)}).collect();
    let v: Vec<_> = vector.iter().enumerate().map(|(i,r)| RetrievalResult{id:r.id.clone(),score:1.0/(k+i as f32+1.0)}).collect();
    weighted_fusion(&l,&v,1.0,1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_and_normalize_are_deterministic() {
        let c = chunk_markdown("d","Hello, World! Rust.",2);
        assert_eq!(c.len(),2);
        assert_eq!(normalize_tokens("Hello, World!"), vec!["hello","world"]);
    }
    #[test]
    fn rrf_merges_and_orders_results() {
        let lexical = vec![RetrievalResult{id:"a".into(),score:10.0}, RetrievalResult{id:"b".into(),score:9.0}];
        let vector = vec![RetrievalResult{id:"b".into(),score:8.0}, RetrievalResult{id:"c".into(),score:7.0}];
        let fused = reciprocal_rank_fusion(&lexical,&vector,60.0);
        assert_eq!(fused[0].id,"b");
        assert_eq!(fused.len(),3);
    }
}
