//! Maskbolt: Segment Anything (SAM) on CPU for AI agents and scripts.
//!
//! [`session::Session`] holds one image and its masks; a [`backend::Backend`] does the model work
//! ([`sam::SamBackend`] with candle, or [`backend::FakeBackend`] for tests). [`mcp`] is the stdio MCP server.
pub mod backend;
pub mod export;
pub mod mask;
pub mod mcp;
pub mod models;
pub mod outline;
pub mod paths;
pub mod sam;
mod sam_decoder;
pub mod session;

pub use backend::{Backend, FakeBackend, Point, Prompt, Proposal};
pub use mask::Mask;
pub use session::{ClickArgs, FindArgs, Level, Region, Session};

use std::sync::Arc;

/// Download (first use) and load a SAM model. `progress(done, total)` reports download bytes.
pub fn load_model(kind: models::ModelKind, progress: impl FnMut(u64, u64)) -> anyhow::Result<(Arc<dyn Backend>, models::Fetched)> {
    let spec = models::spec(kind);
    let mut progress = progress;
    let fetched = models::ensure(spec, &mut progress)?;
    match sam::SamBackend::load(kind, std::path::Path::new(&fetched.path)) {
        Ok(b) => Ok((Arc::new(b), fetched)),
        Err(first) => {
            // A converted file that will not load is rebuilt once from the checked checkpoint.
            let _ = std::fs::remove_file(models::sidecar_path(spec));
            let _ = std::fs::remove_file(models::converted_path(spec));
            let fetched = models::ensure(spec, &mut progress)?;
            let b = sam::SamBackend::load(kind, std::path::Path::new(&fetched.path)).map_err(|e| e.context(format!("loading the model failed twice (first: {first:#})")))?;
            Ok((Arc::new(b), fetched))
        }
    }
}
