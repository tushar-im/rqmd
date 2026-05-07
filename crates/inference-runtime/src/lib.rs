use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type RuntimeResult<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(pub Vec<f32>);

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredDocument {
    pub id: String,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedText {
    pub text: String,
}

pub trait Embedder: Send + Sync {
    fn embed(&self, input: &str) -> RuntimeResult<Embedding>;
}
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, candidates: &[String]) -> RuntimeResult<Vec<ScoredDocument>>;
}
pub trait Expander: Send + Sync {
    fn expand(&self, query: &str) -> RuntimeResult<Vec<String>>;
}
pub trait Generator: Send + Sync {
    fn generate(&self, query: &str, context: &[String], max_tokens: usize) -> RuntimeResult<GeneratedText>;
}

#[derive(Default)]
pub struct RuntimeRegistry {
    pub embedder: Option<Arc<dyn Embedder>>,
    pub reranker: Option<Arc<dyn Reranker>>,
    pub expander: Option<Arc<dyn Expander>>,
    pub generator: Option<Arc<dyn Generator>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum QualityPreset { Small, BalancedQuality }

#[derive(Clone, Debug)]
pub struct ModelManifest {
    pub model_name: String,
    pub checksum: String,
    pub quantized: bool,
}

#[derive(Default)]
pub struct ModelCache {
    pub root: PathBuf,
    pub loaded: HashMap<String, ModelManifest>,
}

impl ModelCache {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self { root: root.as_ref().to_path_buf(), loaded: HashMap::new() }
    }

    pub fn ensure_layout(&self) -> RuntimeResult<()> {
        fs::create_dir_all(self.root.join("manifests")).map_err(|e| e.to_string())?;
        fs::create_dir_all(self.root.join("weights")).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn quantized_by_default(preset: QualityPreset) -> bool {
        matches!(preset, QualityPreset::Small | QualityPreset::BalancedQuality)
    }
}

pub fn adaptive_should_rerank(fused_top_score: f32, agreement_ratio: f32) -> bool {
    !(fused_top_score > 0.92 && agreement_ratio > 0.8)
}

pub fn capped_max_tokens(requested: usize, cap: usize) -> usize { requested.min(cap) }

#[cfg(test)]
mod tests {
    use super::*;
    struct NoopEmbedder;
    impl Embedder for NoopEmbedder {
        fn embed(&self, input: &str) -> RuntimeResult<Embedding> { Ok(Embedding(vec![input.len() as f32])) }
    }

    #[test]
    fn registry_can_store_embedder() {
        let mut r = RuntimeRegistry::default();
        r.embedder = Some(Arc::new(NoopEmbedder));
        let vec = r.embedder.expect("set").embed("hello").expect("ok");
        assert_eq!(vec, Embedding(vec![5.0]));
    }

    #[test]
    fn cache_layout_is_created() {
        let root = std::env::temp_dir().join("qmd-rs-test-cache");
        let cache = ModelCache::new(&root);
        cache.ensure_layout().expect("layout");
        assert!(root.join("manifests").exists());
    }
}
