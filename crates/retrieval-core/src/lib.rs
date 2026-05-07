use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentRecord {
    pub id: String,
    pub title: Option<String>,
    pub body: String,
    pub metadata: HashMap<String, String>,
}

pub fn parse_markdown_record(doc_id: &str, content: &str) -> DocumentRecord {
    let mut lines = content.lines();
    let first = lines.next().unwrap_or_default().trim();
    let title = first.strip_prefix("# ").map(str::to_string);
    let body = content.trim().to_string();
    let mut metadata = HashMap::new();
    metadata.insert("source_format".into(), "markdown".into());
    metadata.insert(
        "token_count".into(),
        normalize_tokens(&body).len().to_string(),
    );
    DocumentRecord {
        id: doc_id.to_string(),
        title,
        body,
        metadata,
    }
}

pub fn chunk_record(record: &DocumentRecord, chunk_size: usize) -> Vec<DocumentChunk> {
    chunk_markdown(&record.id, &record.body, chunk_size)
}

pub fn chunk_markdown(doc_id: &str, content: &str, chunk_size: usize) -> Vec<DocumentChunk> {
    let words: Vec<&str> = content.split_whitespace().collect();
    words
        .chunks(chunk_size.max(1))
        .enumerate()
        .map(|(i, chunk)| DocumentChunk {
            id: format!("{doc_id}:{i}"),
            text: chunk.join(" "),
        })
        .collect()
}

pub fn normalize_tokens(input: &str) -> Vec<String> {
    input
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

pub trait LexicalRetriever: Send + Sync {
    fn search(&self, query: &str, top_k: usize) -> Vec<RetrievalResult>;
}
pub trait VectorRetriever: Send + Sync {
    fn search(&self, query: &str, top_k: usize) -> Vec<RetrievalResult>;
}

#[derive(Clone, Debug)]
pub struct Bm25Index {
    docs: Vec<DocumentChunk>,
    doc_term_freqs: Vec<HashMap<String, usize>>,
    doc_freqs: HashMap<String, usize>,
    avg_doc_len: f32,
    k1: f32,
    b: f32,
}

impl Bm25Index {
    pub fn new(chunks: Vec<DocumentChunk>) -> Self {
        Self::with_params(chunks, 1.2, 0.75)
    }
    pub fn documents(&self) -> &[DocumentChunk] {
        &self.docs
    }
    pub fn with_params(chunks: Vec<DocumentChunk>, k1: f32, b: f32) -> Self {
        let mut doc_term_freqs = Vec::with_capacity(chunks.len());
        let mut doc_freqs: HashMap<String, usize> = HashMap::new();
        let mut total_doc_len = 0usize;

        for chunk in &chunks {
            let tokens = normalize_tokens(&chunk.text);
            total_doc_len += tokens.len();
            let mut tf: HashMap<String, usize> = HashMap::new();
            for token in tokens {
                *tf.entry(token).or_default() += 1;
            }
            for term in tf.keys() {
                *doc_freqs.entry(term.clone()).or_default() += 1;
            }
            doc_term_freqs.push(tf);
        }

        let avg_doc_len = if chunks.is_empty() {
            0.0
        } else {
            total_doc_len as f32 / chunks.len() as f32
        };
        Self {
            docs: chunks,
            doc_term_freqs,
            doc_freqs,
            avg_doc_len,
            k1,
            b,
        }
    }

    fn idf(&self, term: &str) -> f32 {
        let n_docs = self.docs.len() as f32;
        let n_q = *self.doc_freqs.get(term).unwrap_or(&0) as f32;
        ((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0).ln()
    }

    fn score_doc(&self, query_terms: &[String], doc_idx: usize) -> f32 {
        if self.docs.is_empty() {
            return 0.0;
        }
        let tf = &self.doc_term_freqs[doc_idx];
        let doc_len = tf.values().sum::<usize>() as f32;
        query_terms.iter().fold(0.0, |acc, term| {
            let f_qd = *tf.get(term).unwrap_or(&0) as f32;
            if f_qd <= 0.0 {
                return acc;
            }
            let idf = self.idf(term);
            let denom =
                f_qd + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len.max(1e-6)));
            acc + idf * ((f_qd * (self.k1 + 1.0)) / denom)
        })
    }
}

impl LexicalRetriever for Bm25Index {
    fn search(&self, query: &str, top_k: usize) -> Vec<RetrievalResult> {
        let query_terms = normalize_tokens(query);
        let mut scored: Vec<RetrievalResult> = self
            .docs
            .iter()
            .enumerate()
            .map(|(i, doc)| RetrievalResult {
                id: doc.id.clone(),
                score: self.score_doc(&query_terms, i),
            })
            .filter(|result| result.score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        scored.into_iter().take(top_k).collect()
    }
}

#[derive(Clone, Debug)]
pub struct AnnIndex {
    docs: Vec<DocumentChunk>,
    doc_terms: Vec<HashSet<String>>,
}

impl AnnIndex {
    pub fn new(chunks: &[DocumentChunk]) -> Self {
        let docs = chunks.to_vec();
        let doc_terms = docs
            .iter()
            .map(|d| {
                normalize_tokens(&d.text)
                    .into_iter()
                    .collect::<HashSet<_>>()
            })
            .collect();
        Self { docs, doc_terms }
    }

    fn jaccard_score(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
        let inter = a.intersection(b).count() as f32;
        let union = a.union(b).count() as f32;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

impl VectorRetriever for AnnIndex {
    fn search(&self, query: &str, top_k: usize) -> Vec<RetrievalResult> {
        let q: HashSet<String> = normalize_tokens(query).into_iter().collect();
        let mut out: Vec<RetrievalResult> = self
            .docs
            .iter()
            .zip(self.doc_terms.iter())
            .map(|(d, terms)| RetrievalResult {
                id: d.id.clone(),
                score: Self::jaccard_score(&q, terms),
            })
            .filter(|r| r.score > 0.0)
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        out.into_iter().take(top_k).collect()
    }
}

pub fn weighted_fusion(
    lexical: &[RetrievalResult],
    vector: &[RetrievalResult],
    lw: f32,
    vw: f32,
) -> Vec<RetrievalResult> {
    let mut acc: HashMap<&str, f32> = HashMap::new();
    for r in lexical {
        *acc.entry(&r.id).or_default() += r.score * lw;
    }
    for r in vector {
        *acc.entry(&r.id).or_default() += r.score * vw;
    }
    let mut out: Vec<RetrievalResult> = acc
        .into_iter()
        .map(|(id, score)| RetrievalResult {
            id: id.into(),
            score,
        })
        .collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    out
}

pub fn reciprocal_rank_fusion(
    lexical: &[RetrievalResult],
    vector: &[RetrievalResult],
    k: f32,
) -> Vec<RetrievalResult> {
    let l: Vec<_> = lexical
        .iter()
        .enumerate()
        .map(|(i, r)| RetrievalResult {
            id: r.id.clone(),
            score: 1.0 / (k + i as f32 + 1.0),
        })
        .collect();
    let v: Vec<_> = vector
        .iter()
        .enumerate()
        .map(|(i, r)| RetrievalResult {
            id: r.id.clone(),
            score: 1.0 / (k + i as f32 + 1.0),
        })
        .collect();
    weighted_fusion(&l, &v, 1.0, 1.0)
}

pub fn build_retrieval_indices(chunks: Vec<DocumentChunk>) -> (Bm25Index, AnnIndex) {
    let bm25 = Bm25Index::new(chunks);
    let ann = AnnIndex::new(bm25.documents());
    (bm25, ann)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_and_normalize_are_deterministic() {
        let c = chunk_markdown("d", "Hello, World! Rust.", 2);
        assert_eq!(c.len(), 2);
        assert_eq!(normalize_tokens("Hello, World!"), vec!["hello", "world"]);
    }

    #[test]
    fn bm25_ranks_exact_term_match_highest() {
        let index = Bm25Index::new(vec![
            DocumentChunk {
                id: "a".into(),
                text: "rust retrieval fusion".into(),
            },
            DocumentChunk {
                id: "b".into(),
                text: "python scripts".into(),
            },
            DocumentChunk {
                id: "c".into(),
                text: "rust rust memory".into(),
            },
        ]);
        let results = index.search("rust", 3);
        assert_eq!(results[0].id, "c");
        assert_eq!(results[1].id, "a");
    }

    #[test]
    fn parse_markdown_extracts_title_and_metadata() {
        let record = parse_markdown_record("doc-1", "# Intro\nRust retrieval core.");
        assert_eq!(record.title.as_deref(), Some("Intro"));
        assert_eq!(
            record.metadata.get("source_format").map(String::as_str),
            Some("markdown")
        );
        let chunks = chunk_record(&record, 2);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn ann_and_bm25_can_be_fused() {
        let chunks = vec![
            DocumentChunk {
                id: "a".into(),
                text: "rust retrieval fusion".into(),
            },
            DocumentChunk {
                id: "b".into(),
                text: "python scripts".into(),
            },
            DocumentChunk {
                id: "c".into(),
                text: "rust memory safety".into(),
            },
        ];
        let (bm25, ann) = build_retrieval_indices(chunks);
        let fused = reciprocal_rank_fusion(&bm25.search("rust", 3), &ann.search("rust", 3), 60.0);
        assert_eq!(fused[0].id, "a");
    }
}
