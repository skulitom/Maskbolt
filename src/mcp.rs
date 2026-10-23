//! The stdio MCP server: eight tools over one open image. Every reply that changes masks says how many there
//! are, why candidates were dropped, and what still looks uncovered; find and view replies carry a numbered PNG.
//! Paths are confined to the root folder.
use crate::backend::{Backend, Point};
use crate::export::{self, ExportArgs, Format};
use crate::models::{self, ModelKind};
use crate::paths::Root;
use crate::session::{ClickArgs, FindArgs, Level, Region, Session};
use anyhow::{bail, ensure, Result};
use base64::Engine;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig},
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router, ErrorData, ServerHandler, ServiceExt,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

enum ModelState {
    Idle,
    Downloading { done: u64, total: u64, started: Instant },
    Ready(Arc<dyn Backend>),
    Failed(String),
}

struct Inner {
    model: ModelState,
    session: Option<Session>,
    /// Set when weights were downloaded; the next open reply reports it once.
    downloaded: Option<Value>,
}

#[derive(Clone)]
pub struct Maskbolt {
    root: Root,
    kind: ModelKind,
    inner: Arc<Mutex<Inner>>,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct OpenParams {
    /// Image file (PNG, JPEG or WebP) inside the root folder; relative paths are resolved against the root.
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindParams {
    /// Prompt points per side (default 16; 32 for dense scenes; per tile when tiling; at most 48).
    pub grid: Option<u32>,
    /// Only look inside this region of the image (pixels).
    pub region: Option<Region>,
    /// Tile size in pixels (e.g. 400): overlapping crops, each re-encoded at full model resolution, so small
    /// or dense objects show up. Slower: one encode per tile; at most 64 tiles per call.
    pub tile: Option<u32>,
    /// Minimum predicted IoU (default 0.86).
    pub min_score: Option<f32>,
    /// Smallest mask as a fraction of the image area (default 0.0005).
    pub min_area_frac: Option<f64>,
    /// Largest mask as a fraction of the image area (default 0.25).
    pub max_area_frac: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClickParams {
    /// Points on the object, [[x, y], ...] in image pixels; a third value 0 marks a point NOT on the object.
    pub points: Option<Vec<Vec<f32>>>,
    /// Points that are not part of the object, [[x, y], ...].
    pub exclude: Option<Vec<[f32; 2]>>,
    /// Box around the object [x0, y0, x1, y1]; alone, its centre is used as the point.
    #[serde(rename = "box")]
    pub bbox: Option<[f32; 4]>,
    /// best (default: highest predicted IoU), smallest (a part), middle (the object), largest (with what it holds).
    pub level: Option<Level>,
    /// Keep the mask even if it duplicates or overlaps an existing one.
    pub force: Option<bool>,
    /// Fill holes inside the new mask.
    pub fill_holes: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ViewParams {
    /// Zoom on this region (pixels); small regions are enlarged.
    pub region: Option<Region>,
    /// Only these mask ids.
    pub ids: Option<Vec<u32>>,
    /// Longest side of the returned PNG (default 1000, at most 1568).
    pub max_side: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RemoveParams {
    /// Mask ids to remove.
    pub ids: Vec<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GapsParams {
    /// Click every candidate point now and keep masks of a plausible size (default false: only list them).
    pub add: Option<bool>,
    /// With add: minimum predicted IoU (default 0.8).
    pub min_score: Option<f32>,
    /// With add: rounds of gap search and clicks (default 2).
    pub rounds: Option<u32>,
    /// Most points to list (default 50).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FillHolesParams {
    /// Mask ids (default: all masks).
    pub ids: Option<Vec<u32>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportParams {
    /// overlay (numbered PNG), cutout (transparent PNG), svg (outlines with holes), json (outlines, areas,
    /// boxes, scores) or labels (PNG, one colour per mask).
    pub format: Format,
    /// Output file inside the root folder; missing folders are created.
    pub path: String,
    /// Mask ids (default: all).
    pub ids: Option<Vec<u32>>,
    /// Millimetres per pixel, for svg and json in mm.
    pub mm_per_px: Option<f64>,
    /// overlay only: zoom on a region.
    pub region: Option<Region>,
    /// Outline simplification tolerance in pixels (default 1.2).
    pub simplify_px: Option<f64>,
    /// Replace the file if it already exists (default false). The open image is never replaced.
    pub overwrite: Option<bool>,
}

fn text(v: Value) -> ContentBlock {
    ContentBlock::text(v.to_string())
}

fn png(bytes: &[u8]) -> ContentBlock {
    ContentBlock::image(base64::engine::general_purpose::STANDARD.encode(bytes), "image/png")
}

fn fail(e: anyhow::Error) -> CallToolResult {
    CallToolResult::error(vec![text(json!({"error": format!("{e:#}")}))])
}

impl Maskbolt {
    /// A server that loads the SAM model (downloading it on first use).
    pub fn new(root: &Path, kind: ModelKind) -> Result<Self> {
        Ok(Self { root: Root::for_server(root)?, kind, inner: Arc::new(Mutex::new(Inner { model: ModelState::Idle, session: None, downloaded: None })), tool_router: Self::tool_router() })
    }

    /// A server over a given backend (tests, other models).
    pub fn with_backend(root: &Path, backend: Arc<dyn Backend>) -> Result<Self> {
        let s = Self::new(root, ModelKind::VitB)?;
        s.inner.lock().unwrap_or_else(|e| e.into_inner()).model = ModelState::Ready(backend);
        Ok(s)
    }

    /// Run blocking work on the open session.
    async fn with_session<F>(&self, f: F) -> Result<CallToolResult, ErrorData>
    where
        F: FnOnce(&mut Session, &Root) -> Result<Vec<ContentBlock>> + Send + 'static,
    {
        let inner = self.inner.clone();
        let root = self.root.clone();
        let res = tokio::task::spawn_blocking(move || -> Result<Vec<ContentBlock>> {
            let mut g = inner.lock().unwrap_or_else(|e| e.into_inner());
            let Some(s) = g.session.as_mut() else { bail!("no image is open: call maskbolt_open first") };
            f(s, &root)
        })
        .await
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(match res {
            Ok(c) => CallToolResult::success(c),
            Err(e) => fail(e),
        })
    }

    /// Get the backend, starting the first-use download in the background if needed. Returns Ok(None) with a
    /// status reply while weights are still downloading.
    fn backend(&self) -> Result<std::result::Result<Arc<dyn Backend>, Value>> {
        let spec = models::spec(self.kind);
        {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            match &g.model {
                ModelState::Ready(b) => return Ok(Ok(b.clone())),
                ModelState::Failed(e) => {
                    let e = e.clone();
                    g.model = ModelState::Idle;
                    bail!("model download or load failed: {e} (call maskbolt_open again to retry)");
                }
                ModelState::Downloading { .. } => {}
                ModelState::Idle => {
                    if models::is_ready(spec) {
                        let (b, _) = crate::load_model(self.kind, |_, _| {})?;
                        g.model = ModelState::Ready(b.clone());
                        return Ok(Ok(b));
                    }
                    g.model = ModelState::Downloading { done: 0, total: spec.bytes, started: Instant::now() };
                    let inner = self.inner.clone();
                    let kind = self.kind;
                    std::thread::spawn(move || {
                        let r = crate::load_model(kind, |done, total| {
                            {
                                let mut g = inner.lock().unwrap_or_else(|e| e.into_inner());
                                if let ModelState::Downloading { done: d, total: t, .. } = &mut g.model {
                                    *d = done;
                                    *t = total;
                                }
                            }
                        });
                        let mut g = inner.lock().unwrap_or_else(|e| e.into_inner());
                        g.model = match r {
                            Ok((b, f)) => {
                                if f.downloaded {
                                    g.downloaded = Some(json!({"model": spec.name, "size_mb": (f.bytes as f64 / 1e6).round(), "source": f.url,
                                        "sha256_checked": f.sha256, "licence": f.licence, "saved_as": f.path}));
                                }
                                ModelState::Ready(b)
                            }
                            Err(e) => ModelState::Failed(format!("{e:#}")),
                        };
                    });
                }
            }
        }
        // Wait a little: on a fast connection the first open completes in one call.
        let until = Instant::now() + Duration::from_secs(25);
        loop {
            {
                let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                match &g.model {
                    ModelState::Ready(b) => return Ok(Ok(b.clone())),
                    ModelState::Failed(e) => {
                        let e = e.clone();
                        drop(g);
                        self.inner.lock().unwrap_or_else(|p| p.into_inner()).model = ModelState::Idle;
                        bail!("model download or load failed: {e} (call maskbolt_open again to retry)");
                    }
                    ModelState::Downloading { done, total, started } => {
                        if Instant::now() >= until {
                            return Ok(Err(json!({
                                "status": "downloading model",
                                "model": spec.name,
                                "size_mb": (spec.bytes as f64 / 1e6).round(),
                                "done_mb": (*done as f64 / 1e6).round(),
                                "total_mb": (*total as f64 / 1e6).round(),
                                "seconds": started.elapsed().as_secs(),
                                "source": spec.url,
                                "licence": spec.licence,
                                "sha256": spec.sha256,
                                "folder": models::cache_dir().display().to_string(),
                                "hint": format!("First use only: the {} weights ({:.0} MB, {}) are downloading and will be SHA-256 checked. Call maskbolt_open again in a minute.", spec.name, spec.bytes as f64 / 1e6, spec.licence)
                            })));
                        }
                    }
                    ModelState::Idle => bail!("model not loaded"),
                }
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}

fn status(s: &Session) -> Value {
    json!({"masks": s.items.len(), "still_uncovered": s.uncovered()})
}

/// A zoom around one mask for click replies.
fn around(s: &Session, id: u32) -> Option<Region> {
    let m = s.items.iter().find(|m| m.id == id)?;
    let b = m.mask.bbox();
    let (w, h) = s.size();
    let pad = ((b[2] - b[0]).max(b[3] - b[1]) / 4 + 16) as i64;
    let x0 = (b[0] as i64 - pad).max(0) as u32;
    let y0 = (b[1] as i64 - pad).max(0) as u32;
    let x1 = ((b[2] as i64 + pad) as u32).min(w - 1);
    let y1 = ((b[3] as i64 + pad) as u32).min(h - 1);
    Some(Region { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 })
}

#[tool_router]
impl Maskbolt {
    #[tool(description = "Open an image and encode it with Segment Anything (about 7 s on CPU; the first call also downloads the model once and says its size). Clears all masks. Use this first, and again for each new image. Path must be inside the server's root folder.")]
    async fn maskbolt_open(&self, Parameters(p): Parameters<OpenParams>) -> Result<CallToolResult, ErrorData> {
        let this = self.clone();
        let res = tokio::task::spawn_blocking(move || -> Result<Vec<ContentBlock>> {
            let path = this.root.read(&p.path)?;
            let backend = match this.backend()? {
                Ok(b) => b,
                Err(status) => return Ok(vec![text(status)]),
            };
            let mut s = Session::new(backend);
            let info = s.open(&path)?;
            let mut g = this.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.session = Some(s);
            let mut v = json!({"opened": info, "masks": 0,
                "next": "maskbolt_click on one object, or maskbolt_find for every object (add tile for small or dense objects); then maskbolt_view to check."});
            if let Some(d) = g.downloaded.take() {
                v["model_download"] = d;
            }
            Ok(vec![text(v)])
        })
        .await
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(match res {
            Ok(c) => CallToolResult::success(c),
            Err(e) => fail(e),
        })
    }

    #[tool(description = "Find every object automatically with a grid of point prompts, over the whole image, a region, or overlapping tiles (tile: e.g. 400 px; each tile is re-encoded at full model resolution, so small or dense objects show up; masks cut by a tile edge are dropped, the overlap covers them). Duplicates and masks inside or containing existing ones are skipped. A tiled find does not run a whole-image pass: objects larger than about a third of a tile are dropped, so run a find without tile first for big objects. At most 64 tiles and a 48x48 grid per call (about 3 s per tile and 0.02 s per prompt on a 16-core CPU). Returns counts, why candidates were dropped, what still looks uncovered, and a numbered preview. Use it after open for 'all the objects' tasks; then view, remove wrong masks, click missed ones.")]
    async fn maskbolt_find(&self, Parameters(p): Parameters<FindParams>) -> Result<CallToolResult, ErrorData> {
        self.with_session(move |s, _| {
            let d = FindArgs::default();
            let a = FindArgs {
                grid: p.grid.unwrap_or(d.grid),
                region: p.region,
                tile: p.tile,
                min_score: p.min_score.unwrap_or(d.min_score),
                min_area_frac: p.min_area_frac.unwrap_or(d.min_area_frac),
                max_area_frac: p.max_area_frac.unwrap_or(d.max_area_frac),
            };
            let rep = s.find(&a)?;
            let prev = s.preview(None, 1000, None)?;
            Ok(vec![text(json!({"find": rep, "status": status(s), "preview": {"scale": prev.scale}})), png(&prev.png)])
        })
        .await
    }

    #[tool(description = "Segment one object from points (on it; a third value 0, or 'exclude', marks points not on it) and/or a box [x0,y0,x1,y1], in image pixels. Picks SAM's best mask, or level smallest (a part), middle (the object), largest (the object with what it holds); the reply lists all three sizes so you can switch. Skips a mask that duplicates an existing one, lies inside one ('inside #n') or contains one ('contains #n') unless force is true. Returns the new id, what is still uncovered, and a zoomed preview. Use it for single cutouts and to add objects find missed.")]
    async fn maskbolt_click(&self, Parameters(p): Parameters<ClickParams>) -> Result<CallToolResult, ErrorData> {
        self.with_session(move |s, _| {
            let mut pts = Vec::new();
            for q in p.points.unwrap_or_default() {
                ensure!(q.len() == 2 || q.len() == 3, "each point is [x, y] or [x, y, label]");
                pts.push(Point { x: q[0], y: q[1], include: q.get(2).is_none_or(|l| *l != 0.0) });
            }
            for q in p.exclude.unwrap_or_default() {
                pts.push(Point { x: q[0], y: q[1], include: false });
            }
            let r = s.click(&ClickArgs { points: pts, bbox: p.bbox, level: p.level.unwrap_or_default(), force: p.force.unwrap_or(false), fill_holes: p.fill_holes.unwrap_or(false) })?;
            let mut out = vec![text(json!({"click": r, "status": status(s)}))];
            if let Some(region) = r.id.and_then(|id| around(s, id)) {
                out.push(png(&s.preview(Some(region), 600, None)?.png));
            }
            Ok(out)
        })
        .await
    }

    #[tool(description = "Show the image with every mask filled in its own colour, outlined, with its id at its centre, plus the list of masks (id, area, bbox, score, source). Pass region {x,y,w,h} to zoom (small regions are enlarged): use zooms to check small or dense objects before removing or clicking. Preview pixels map to image pixels as image_x = region.x + preview_x / scale (same for y).")]
    async fn maskbolt_view(&self, Parameters(p): Parameters<ViewParams>) -> Result<CallToolResult, ErrorData> {
        self.with_session(move |s, _| {
            let list = s.summary(p.region, p.ids.as_deref());
            let shown = 300;
            let prev = s.preview(p.region, p.max_side.unwrap_or(1000).clamp(64, 1568), p.ids.as_deref())?;
            Ok(vec![
                text(json!({"total": s.items.len(), "listed": list.len().min(shown), "more": list.len().saturating_sub(shown),
                    "masks": &list[..list.len().min(shown)], "preview": {"region": prev.region, "scale": prev.scale, "width": prev.width, "height": prev.height,
                        "to_image": "image_x = region.x + preview_x / scale, image_y = region.y + preview_y / scale"}})),
                png(&prev.png),
            ])
        })
        .await
    }

    #[tool(description = "Remove masks by id (wrong, merged or duplicate ones seen in view). Returns what was removed and what is still uncovered.")]
    async fn maskbolt_remove(&self, Parameters(p): Parameters<RemoveParams>) -> Result<CallToolResult, ErrorData> {
        self.with_session(move |s, _| {
            let (removed, missing) = s.remove(&p.ids);
            Ok(vec![text(json!({"removed": removed, "not_found": missing, "status": status(s)}))])
        })
        .await
    }

    #[tool(description = "Find more like these: list uncovered patches whose colour matches the masks found so far (at least 0.3x the median mask size), with one click point each, biggest first. With add: true, clicks them all and keeps masks of a plausible size (0.3 to 2.5x the median), in up to 2 rounds. Use it when find or your clicks left objects out.")]
    async fn maskbolt_gaps(&self, Parameters(p): Parameters<GapsParams>) -> Result<CallToolResult, ErrorData> {
        self.with_session(move |s, _| {
            ensure!(!s.items.is_empty(), "no masks yet: gaps compares against existing masks; use find or click first");
            let limit = p.limit.unwrap_or(50);
            if p.add.unwrap_or(false) {
                let before = s.items.len();
                let rounds = s.fill_gaps(p.min_score.unwrap_or(0.8), p.rounds.unwrap_or(2).clamp(1, 5) as usize)?;
                Ok(vec![text(json!({"added": s.items.len() - before, "rounds": rounds, "status": status(s)}))])
            } else {
                let g = s.gaps(0.3, 2.2);
                Ok(vec![text(json!({"candidates": g.len(), "points": &g[..g.len().min(limit)],
                    "hint": "Click these points (maskbolt_click) or call maskbolt_gaps with add: true."}))])
            }
        })
        .await
    }

    #[tool(description = "Fill holes inside masks (pixels the outside cannot reach without crossing the mask), e.g. a cup whose highlight was left out. Default: all masks. Do not use it where holes are real (rings, plates with cut-outs).")]
    async fn maskbolt_fill_holes(&self, Parameters(p): Parameters<FillHolesParams>) -> Result<CallToolResult, ErrorData> {
        self.with_session(move |s, _| {
            let r = s.fill_holes(p.ids.as_deref())?;
            let filled: Vec<Value> = r.iter().filter(|x| x.1 > 0).map(|(id, f, a)| json!({"id": id, "filled": f, "area": a})).collect();
            Ok(vec![text(json!({"masks_changed": filled.len(), "filled": filled, "status": status(s)}))])
        })
        .await
    }

    #[tool(description = "Write masks to a file inside the root: overlay (numbered PNG; region zooms), cutout (transparent PNG of the chosen masks), svg (outer loops and holes, fill-rule evenodd; mm_per_px gives millimetres), json (outlines with holes, area, bbox, score per mask) or labels (PNG, one colour per mask). Default: all masks. Existing files are kept unless overwrite is true; the open image is never overwritten. Use it when the masks are right.")]
    async fn maskbolt_export(&self, Parameters(p): Parameters<ExportParams>) -> Result<CallToolResult, ErrorData> {
        self.with_session(move |s, root| {
            let path = root.write(&p.path)?;
            let mut a = ExportArgs::new(p.format);
            a.ids = p.ids;
            a.mm_per_px = p.mm_per_px;
            a.region = p.region;
            a.overwrite = p.overwrite.unwrap_or(false);
            if let Some(e) = p.simplify_px {
                ensure!((0.0..=20.0).contains(&e), "simplify_px must be 0 to 20");
                a.simplify_px = e;
            }
            let r = export::export(s, &a, &path)?;
            Ok(vec![text(json!({"exported": r}))])
        })
        .await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Maskbolt {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("maskbolt", env!("CARGO_PKG_VERSION")))
            .with_instructions(format!(
                "Maskbolt segments objects in images with Segment Anything on CPU. Files must be inside {}. \
                 Loop: maskbolt_open; then maskbolt_click for one object, or maskbolt_find for all (tile: 400 for small or dense objects); \
                 maskbolt_view (zoom with region) to check the numbered masks; maskbolt_remove wrong ones; maskbolt_click missed ones; \
                 maskbolt_gaps (add: true) when replies say patches are still uncovered; maskbolt_fill_holes for unwanted holes; \
                 repeat until every object has exactly one mask, then maskbolt_export. Coordinates are image pixels.",
                self.root.dir().display()
            ))
    }
}

/// Serve on stdin/stdout until the client disconnects.
pub async fn serve_stdio(root: &Path, kind: ModelKind) -> Result<()> {
    let server = Maskbolt::new(root, kind)?;
    eprintln!("maskbolt mcp {}: serving files inside {}", env!("CARGO_PKG_VERSION"), server.root.dir().display());
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
