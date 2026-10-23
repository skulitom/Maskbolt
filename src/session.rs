//! One open image and its masks: click, find (grid prompts over the image or over overlapping tiles), dedupe,
//! remove, fill holes, gaps ("find more like these"), previews. Ported from the 2026-10-23 Node prototype.
use crate::backend::{Backend, Embedding, Point, Prompt, Proposal};
use crate::mask::Mask;
use anyhow::{bail, ensure, Context, Result};
use image::{RgbImage, RgbaImage};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Which of SAM's three nested masks to keep: the one with the best predicted IoU (default), or by size:
/// smallest (a part), middle (the object), largest (the object with what it sits in or holds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    #[default]
    Best,
    Smallest,
    Middle,
    Largest,
}

impl std::str::FromStr for Level {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "best" => Level::Best,
            "smallest" | "part" => Level::Smallest,
            "middle" | "object" => Level::Middle,
            "largest" | "group" => Level::Largest,
            _ => bail!("level must be best, smallest, middle or largest"),
        })
    }
}

/// A rectangle in image pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Region {
    fn clamp_to(self, w: u32, h: u32) -> Result<Region> {
        ensure!(self.x < w && self.y < h, "region starts outside the image ({w}x{h})");
        let r = Region { x: self.x, y: self.y, w: self.w.min(w - self.x), h: self.h.min(h - self.y) };
        ensure!(r.w > 0 && r.h > 0, "region is empty");
        Ok(r)
    }
    fn overlaps(&self, b: [u32; 4]) -> bool {
        b[2] >= self.x && b[0] < self.x.saturating_add(self.w) && b[3] >= self.y && b[1] < self.y.saturating_add(self.h)
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub id: u32,
    pub mask: Mask,
    pub score: f32,
    pub source: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct MaskInfo {
    pub id: u32,
    pub area: u64,
    pub bbox: [u32; 4],
    pub score: f32,
    pub source: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenInfo {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub encode_ms: u64,
    pub model: String,
}

#[derive(Clone, Debug, Default)]
pub struct ClickArgs {
    pub points: Vec<Point>,
    pub bbox: Option<[f32; 4]>,
    pub level: Level,
    pub force: bool,
    pub fill_holes: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LevelOption {
    pub level: Level,
    pub area: u64,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClickResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
    pub level: Level,
    pub score: f32,
    pub area: u64,
    pub bbox: [u32; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holes_filled: Option<u64>,
    /// SAM's three nested masks for this prompt, smallest first, so the caller can pick another level.
    pub levels: Vec<LevelOption>,
    pub ms: u64,
}

#[derive(Clone, Debug)]
pub struct FindArgs {
    /// Points per side of the prompt grid (per tile when tiling).
    pub grid: u32,
    /// Limit to this part of the image.
    pub region: Option<Region>,
    /// Tile size in pixels: overlapping square crops (stride 60% of the tile), each re-encoded at full model
    /// resolution, so small objects show up.
    pub tile: Option<u32>,
    pub min_score: f32,
    pub min_area_frac: f64,
    pub max_area_frac: f64,
}

impl Default for FindArgs {
    fn default() -> Self {
        Self { grid: 16, region: None, tile: None, min_score: 0.86, min_area_frac: 0.0005, max_area_frac: 0.25 }
    }
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct FindReport {
    pub prompts: usize,
    pub crops: usize,
    pub kept: usize,
    pub total: usize,
    /// Why candidates were not kept, with counts.
    pub dropped: BTreeMap<String, u64>,
    pub encode_s: f64,
    pub decode_s: f64,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Gap {
    pub x: u32,
    pub y: u32,
    pub area: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct GapRound {
    pub round: usize,
    pub gaps: usize,
    pub kept: usize,
    pub total: usize,
    pub dropped: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Uncovered {
    pub patches: usize,
    pub area: u64,
    pub top: Vec<Gap>,
    pub hint: String,
}

pub struct Preview {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Output pixels per image pixel.
    pub scale: f64,
    pub region: Region,
}

pub struct Opened {
    pub path: PathBuf,
    pub rgb: RgbImage,
}

pub struct Session {
    backend: Arc<dyn Backend>,
    image: Option<Opened>,
    pub items: Vec<Item>,
    next_id: u32,
    cache: HashMap<Region, Arc<Embedding>>,
    /// Prompts per decoder call (bounds memory: each prompt yields three masks).
    pub batch: usize,
}

impl Session {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend, image: None, items: Vec::new(), next_id: 1, cache: HashMap::new(), batch: 64 }
    }

    pub fn backend_name(&self) -> String {
        self.backend.name()
    }

    pub fn is_open(&self) -> bool {
        self.image.is_some()
    }

    pub fn image(&self) -> Result<&Opened> {
        self.image.as_ref().context("no image is open: call open first")
    }

    pub fn size(&self) -> (u32, u32) {
        self.image.as_ref().map_or((0, 0), |i| i.rgb.dimensions())
    }

    /// Open an image file (PNG, JPEG or WebP) and encode it. Clears all masks.
    pub fn open(&mut self, path: &Path) -> Result<OpenInfo> {
        let img = read_image(path)?;
        self.open_rgb(img, path.to_path_buf())
    }

    pub fn open_rgb(&mut self, rgb: RgbImage, path: PathBuf) -> Result<OpenInfo> {
        let (w, h) = rgb.dimensions();
        ensure!(w > 0 && h > 0, "empty image");
        self.image = Some(Opened { path: path.clone(), rgb });
        self.items.clear();
        self.next_id = 1;
        self.cache.clear();
        let t = Instant::now();
        self.embedding(None)?;
        Ok(OpenInfo { path: path.display().to_string(), width: w, height: h, encode_ms: t.elapsed().as_millis() as u64, model: self.backend.name() })
    }

    fn full(&self) -> Region {
        let (w, h) = self.size();
        Region { x: 0, y: 0, w, h }
    }

    fn embedding(&mut self, region: Option<Region>) -> Result<Arc<Embedding>> {
        let r = region.unwrap_or_else(|| self.full());
        if let Some(e) = self.cache.get(&r) {
            return Ok(e.clone());
        }
        let img = &self.image()?.rgb;
        let whole = Region { x: 0, y: 0, w: img.width(), h: img.height() };
        let e = if r == whole {
            self.backend.encode(img)?
        } else {
            let crop = image::imageops::crop_imm(img, r.x, r.y, r.w, r.h).to_image();
            self.backend.encode(&crop)?
        };
        let e = Arc::new(e);
        self.cache.insert(r, e.clone());
        Ok(e)
    }

    /// Encode several crops side by side (MASKBOLT_ENCODE_THREADS, default 4): the encoder alone leaves most
    /// cores idle, and each encode holds about 1 GB at its peak.
    pub fn prefetch(&mut self, regions: &[Region]) -> Result<()> {
        let todo: Vec<Region> = regions.iter().copied().filter(|r| !self.cache.contains_key(r)).collect();
        if todo.is_empty() {
            return Ok(());
        }
        let workers = std::env::var("MASKBOLT_ENCODE_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0).unwrap_or(4);
        let img = &self.image()?.rgb;
        let backend = self.backend.clone();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let done: std::sync::Mutex<Vec<(Region, Result<Embedding>)>> = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for _ in 0..workers.min(todo.len()) {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&r) = todo.get(i) else { break };
                    let crop = image::imageops::crop_imm(img, r.x, r.y, r.w, r.h).to_image();
                    let e = backend.encode(&crop);
                    done.lock().unwrap().push((r, e));
                });
            }
        });
        for (r, e) in done.into_inner().unwrap() {
            self.cache.insert(r, Arc::new(e?));
        }
        Ok(())
    }

    /// Decode prompts given in full-image pixels against the whole image or a crop; masks come back in
    /// full-image pixels.
    pub fn decode(&mut self, prompts: &[Prompt], region: Option<Region>) -> Result<Vec<Proposal>> {
        let r = region.unwrap_or_else(|| self.full());
        let emb = self.embedding(region)?;
        let local: Vec<Prompt> = prompts
            .iter()
            .map(|p| Prompt {
                points: p.points.iter().map(|q| Point { x: q.x - r.x as f32, y: q.y - r.y as f32, include: q.include }).collect(),
                bbox: p.bbox.map(|b| [b[0] - r.x as f32, b[1] - r.y as f32, b[2] - r.x as f32, b[3] - r.y as f32]),
            })
            .collect();
        let mut out = Vec::with_capacity(prompts.len());
        for chunk in local.chunks(self.batch.max(1)) {
            for mut p in self.backend.decode(&emb, chunk)? {
                for m in p.masks.iter_mut() {
                    m.offset(r.x, r.y);
                }
                out.push(p);
            }
        }
        Ok(out)
    }

    fn pick(p: &Proposal, level: Level) -> usize {
        let mut best = 0;
        for j in 1..3 {
            if p.scores[j] > p.scores[best] {
                best = j;
            }
        }
        if level == Level::Best {
            return best;
        }
        let mut order: Vec<usize> = (0..3).filter(|&j| p.masks[j].area > 0).collect();
        order.sort_by_key(|&j| p.masks[j].area);
        if order.is_empty() {
            return best;
        }
        match level {
            Level::Smallest => order[0],
            Level::Largest => order[order.len() - 1],
            _ => order[order.len() / 2],
        }
    }

    /// Add a mask unless it duplicates or swallows an existing one; returns its id or why it was skipped.
    pub fn add(&mut self, mask: Mask, score: f32, source: &str, force: bool) -> std::result::Result<u32, String> {
        if mask.is_empty() {
            return Err("empty".into());
        }
        if !force {
            for m in &self.items {
                let inter = m.mask.intersection(&mask);
                if inter == 0 {
                    continue;
                }
                let (a, b) = (mask.area as f64, m.mask.area as f64);
                let i = inter as f64;
                if i / (a + b - i) > 0.6 {
                    return Err(format!("duplicate of #{}", m.id));
                }
                if i / a > 0.8 {
                    return Err(format!("inside #{}", m.id));
                }
                if i / b > 0.8 {
                    return Err(format!("contains #{}", m.id));
                }
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.items.push(Item { id, mask, score, source: source.into() });
        Ok(id)
    }

    fn check_point(&self, x: f32, y: f32) -> Result<()> {
        let (w, h) = self.size();
        ensure!(x >= 0.0 && y >= 0.0 && x < w as f32 && y < h as f32, "point ({x}, {y}) is outside the image ({w}x{h} pixels)");
        Ok(())
    }

    /// Segment the object at the given points (include/exclude) and/or inside a box.
    pub fn click(&mut self, a: &ClickArgs) -> Result<ClickResult> {
        self.image()?;
        let mut points = a.points.clone();
        for p in &points {
            self.check_point(p.x, p.y)?;
        }
        if let Some(b) = a.bbox {
            ensure!(b[2] > b[0] && b[3] > b[1], "box must be [x0, y0, x1, y1] with x1 > x0 and y1 > y0");
            self.check_point(b[0], b[1])?;
            let (w, h) = self.size();
            ensure!(b[2] <= w as f32 && b[3] <= h as f32, "box ends outside the image ({w}x{h} pixels)");
            if points.is_empty() {
                // A box alone: add its centre as an include point (as the prototype did).
                points.push(Point { x: (b[0] + b[2]) / 2.0, y: (b[1] + b[3]) / 2.0, include: true });
            }
        }
        ensure!(points.iter().any(|p| p.include) || a.bbox.is_some(), "give at least one include point or a box");
        let t = Instant::now();
        let p = self.decode(&[Prompt { points, bbox: a.bbox }], None)?.remove(0);
        let j = Self::pick(&p, a.level);
        // Name the non-empty masks by size exactly as pick() does (empty ones are never picked).
        let mut order: Vec<usize> = (0..3).filter(|&k| p.masks[k].area > 0).collect();
        order.sort_by_key(|&k| p.masks[k].area);
        let levels = order
            .iter()
            .map(|&k| {
                let level = [Level::Smallest, Level::Middle, Level::Largest].into_iter().find(|&l| Self::pick(&p, l) == k).unwrap_or(Level::Middle);
                LevelOption { level, area: p.masks[k].area, score: p.scores[k] }
            })
            .collect();
        let mut mask = p.masks[j].clone();
        let score = p.scores[j];
        let holes = if a.fill_holes { Some(mask.fill_holes()) } else { None };
        let (area, bbox) = (mask.area, mask.bbox());
        let (id, skipped) = match self.add(mask, score, "click", a.force) {
            Ok(id) => (Some(id), None),
            Err(why) => (None, Some(why)),
        };
        Ok(ClickResult { id, skipped, level: a.level, score, area, bbox, holes_filled: holes, levels, ms: t.elapsed().as_millis() as u64 })
    }

    /// Grid prompts over the image, a region, or overlapping tiles.
    pub fn find(&mut self, a: &FindArgs) -> Result<FindReport> {
        self.image()?;
        ensure!((1..=MAX_GRID).contains(&a.grid), "grid must be 1 to {MAX_GRID} points per side");
        let t = Instant::now();
        let (w, h) = self.size();
        let area = match a.region {
            Some(r) => r.clamp_to(w, h)?,
            None => self.full(),
        };
        let mut rep = FindReport::default();
        let tiles: Vec<Region> = match a.tile {
            Some(tile) => {
                ensure!(tile >= 32, "tile must be at least 32 pixels");
                let (tw, th) = (tile.min(area.w), tile.min(area.h));
                let xs = tile_starts(area.x, area.w, tw);
                let ys = tile_starts(area.y, area.h, th);
                ys.iter().flat_map(|&y| xs.iter().map(move |&x| Region { x, y, w: tw, h: th })).collect()
            }
            None => vec![area],
        };
        if tiles.len() > MAX_TILES {
            let tile = a.tile.unwrap_or(0);
            let mut bigger = tile;
            while tile_count(area, bigger) > MAX_TILES {
                bigger += 16;
            }
            let minutes = tiles.len() as f64 * (SECONDS_PER_TILE + (a.grid * a.grid) as f64 * SECONDS_PER_PROMPT) / 60.0;
            bail!(
                "tile {tile} gives {} tiles over this {}x{} area (at most {MAX_TILES} per find): about {:.0} min on a 16-core CPU. Use a larger tile (tile {bigger} gives {} tiles) or a region.",
                tiles.len(),
                area.w,
                area.h,
                minutes.max(1.0),
                tile_count(area, bigger)
            );
        }
        if tiles.len() > 1 {
            let te = Instant::now();
            self.prefetch(&tiles)?;
            rep.encode_s += te.elapsed().as_secs_f64();
        }
        for r in tiles {
            let region = if r == self.full() { None } else { Some(r) };
            self.find_one(a, region, &mut rep)?;
            if region.is_some() {
                // A tile's embedding (4 MB) is not needed once its prompts are decoded.
                self.cache.remove(&r);
            }
        }
        rep.total = self.items.len();
        rep.seconds = t.elapsed().as_secs_f64();
        Ok(rep)
    }

    fn find_one(&mut self, a: &FindArgs, region: Option<Region>, rep: &mut FindReport) -> Result<()> {
        let (w, h) = self.size();
        let r = region.unwrap_or_else(|| self.full());
        let mut prompts = Vec::new();
        for j in 0..a.grid {
            for i in 0..a.grid {
                let x = r.x as f32 + (i as f32 + 0.5) * r.w as f32 / a.grid as f32;
                let y = r.y as f32 + (j as f32 + 0.5) * r.h as f32 / a.grid as f32;
                prompts.push(Prompt { points: vec![Point { x, y, include: true }], bbox: None });
            }
        }
        let te = Instant::now();
        self.embedding(region)?;
        rep.encode_s += te.elapsed().as_secs_f64();
        let td = Instant::now();
        let props = self.decode(&prompts, region)?;
        rep.decode_s += td.elapsed().as_secs_f64();
        rep.prompts += prompts.len();
        rep.crops += 1;
        let total = (w as f64) * (h as f64);
        let mut cands: Vec<(Mask, f32)> = Vec::new();
        for p in props {
            let j = Self::pick(&p, Level::Best);
            let score = p.scores[j];
            let m = p.masks.into_iter().nth(j).unwrap();
            let reason = if score < a.min_score {
                Some("low score")
            } else if (m.area as f64) < a.min_area_frac * total {
                Some("too small")
            } else if (m.area as f64) > a.max_area_frac * total {
                Some("too large")
            } else if region.is_some() && m.area as f64 > 0.35 * (r.w as f64 * r.h as f64) {
                Some("crop background")
            } else if region.is_some() && cut_by_crop(&m, r, w, h) {
                Some("cut by the crop edge")
            } else {
                None
            };
            if let Some(why) = reason {
                *rep.dropped.entry(why.into()).or_default() += 1;
                continue;
            }
            cands.push((m, score));
        }
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let source = if region.is_some() { "find-tile" } else { "find" };
        for (m, score) in cands {
            match self.add(m, score, source, false) {
                Ok(_) => rep.kept += 1,
                Err(why) => *rep.dropped.entry(why.split(' ').next().unwrap_or("skipped").to_string()).or_default() += 1,
            }
        }
        Ok(())
    }

    pub fn remove(&mut self, ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut removed = Vec::new();
        let mut missing = Vec::new();
        for &id in ids {
            if let Some(i) = self.items.iter().position(|m| m.id == id) {
                self.items.remove(i);
                removed.push(id);
            } else {
                missing.push(id);
            }
        }
        (removed, missing)
    }

    /// Remove masks that are the background seen through another mask's hole (e.g. the laser bed through a
    /// cut-out in an offcut): over 90% inside the other mask once its holes are filled, under 25% inside the mask
    /// itself. Returns the removed ids.
    pub fn remove_hole_views(&mut self) -> Vec<u32> {
        let filled: Vec<Mask> = self
            .items
            .iter()
            .map(|m| {
                let mut c = m.mask.clone();
                c.fill_holes();
                c
            })
            .collect();
        let mut drop = Vec::new();
        for (i, a) in self.items.iter().enumerate() {
            for (j, b) in self.items.iter().enumerate() {
                if i == j || filled[j].area == b.mask.area {
                    continue;
                }
                let area = a.mask.area as f64;
                if a.mask.intersection(&filled[j]) as f64 > 0.9 * area && (a.mask.intersection(&b.mask) as f64) < 0.25 * area {
                    drop.push(a.id);
                    break;
                }
            }
        }
        self.remove(&drop);
        drop
    }

    /// Fill interior holes of the given masks (all when `ids` is None). Returns (id, pixels added, new area).
    pub fn fill_holes(&mut self, ids: Option<&[u32]>) -> Result<Vec<(u32, u64, u64)>> {
        if let Some(ids) = ids {
            for id in ids {
                ensure!(self.items.iter().any(|m| m.id == *id), "no mask #{id}");
            }
        }
        let mut out = Vec::new();
        for m in self.items.iter_mut() {
            if ids.is_none_or(|ids| ids.contains(&m.id)) {
                let added = m.mask.fill_holes();
                out.push((m.id, added, m.mask.area));
            }
        }
        Ok(out)
    }

    /// "Find more like these": uncovered patches whose colour is within `k` standard deviations of the masked
    /// pixels' mean colour, at least `min_frac` x the median mask area; one click point per patch (the patch pixel
    /// nearest its centroid, so a crescent's point lands inside it), biggest first.
    pub fn gaps(&self, min_frac: f64, k: f64) -> Vec<Gap> {
        let Some(img) = self.image.as_ref() else { return Vec::new() };
        if self.items.is_empty() {
            return Vec::new();
        }
        let (w, h) = img.rgb.dimensions();
        let n = (w * h) as usize;
        let px = img.rgb.as_raw();
        let mut covered = vec![0u8; n];
        let mut cnt = 0f64;
        let mut sum = [0f64; 3];
        let mut sq = [0f64; 3];
        for m in &self.items {
            for (x, y) in m.mask.pixels() {
                let i = (y * w + x) as usize;
                if covered[i] == 0 {
                    covered[i] = 1;
                    cnt += 1.0;
                    for c in 0..3 {
                        let v = px[i * 3 + c] as f64;
                        sum[c] += v;
                        sq[c] += v * v;
                    }
                }
            }
        }
        let mean: Vec<f64> = sum.iter().map(|s| s / cnt).collect();
        let sd: Vec<f64> = (0..3).map(|c| (sq[c] / cnt - mean[c] * mean[c]).max(1.0).sqrt()).collect();
        let mut like = vec![0u8; n];
        for i in 0..n {
            if covered[i] != 0 {
                continue;
            }
            if (0..3).all(|c| (px[i * 3 + c] as f64 - mean[c]).abs() <= k * sd[c]) {
                like[i] = 1;
            }
        }
        let mut areas: Vec<u64> = self.items.iter().map(|m| m.mask.area).collect();
        areas.sort_unstable();
        let med = areas[areas.len() / 2].max(1) as f64;
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        let mut comp: Vec<u32> = Vec::new();
        for s in 0..n {
            if like[s] == 0 || seen[s] {
                continue;
            }
            comp.clear();
            let mut stack = vec![s as u32];
            seen[s] = true;
            let (mut sx, mut sy) = (0f64, 0f64);
            while let Some(j) = stack.pop() {
                let (x, y) = (j % w, j / w);
                comp.push(j);
                sx += x as f64;
                sy += y as f64;
                let mut visit = |nj: u32| {
                    if !seen[nj as usize] && like[nj as usize] != 0 {
                        seen[nj as usize] = true;
                        stack.push(nj);
                    }
                };
                if x > 0 {
                    visit(j - 1);
                }
                if x + 1 < w {
                    visit(j + 1);
                }
                if y > 0 {
                    visit(j - w);
                }
                if y + 1 < h {
                    visit(j + w);
                }
            }
            let a = comp.len() as f64;
            if a < min_frac * med {
                continue;
            }
            let (cx, cy) = (sx / a, sy / a);
            let best = comp.iter().copied().min_by(|&p, &q| {
                let d = |j: u32| ((j % w) as f64 - cx).powi(2) + ((j / w) as f64 - cy).powi(2);
                d(p).partial_cmp(&d(q)).unwrap()
            });
            let b = best.unwrap();
            out.push(Gap { x: b % w, y: b / w, area: comp.len() as u64 });
        }
        out.sort_by(|a, b| b.area.cmp(&a.area).then(a.y.cmp(&b.y)).then(a.x.cmp(&b.x)));
        out
    }

    /// Click every gap point and keep masks of a plausible size (0.3 to 2.5 x the median), for `rounds` rounds.
    pub fn fill_gaps(&mut self, min_score: f32, rounds: usize) -> Result<Vec<GapRound>> {
        self.image()?;
        let mut report = Vec::new();
        for r in 0..rounds {
            let g = self.gaps(0.3, 2.2);
            if g.is_empty() {
                break;
            }
            let mut areas: Vec<u64> = self.items.iter().map(|m| m.mask.area).collect();
            areas.sort_unstable();
            let med = areas[areas.len() / 2] as f64;
            let prompts: Vec<Prompt> = g.iter().map(|p| Prompt { points: vec![Point { x: p.x as f32, y: p.y as f32, include: true }], bbox: None }).collect();
            let props = self.decode(&prompts, None)?;
            let mut cands: Vec<(Mask, f32)> = props
                .into_iter()
                .map(|p| {
                    let j = Self::pick(&p, Level::Best);
                    let s = p.scores[j];
                    (p.masks.into_iter().nth(j).unwrap(), s)
                })
                .collect();
            cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut kept = 0;
            let mut dropped: BTreeMap<String, u64> = BTreeMap::new();
            for (m, s) in cands {
                let a = m.area as f64;
                let why = if s < min_score {
                    Some("low score")
                } else if a < 0.3 * med {
                    Some("too small")
                } else if a > 2.5 * med {
                    Some("too large")
                } else {
                    None
                };
                if let Some(why) = why {
                    *dropped.entry(why.into()).or_default() += 1;
                    continue;
                }
                match self.add(m, s, "gap", false) {
                    Ok(_) => kept += 1,
                    Err(why) => *dropped.entry(why.split(' ').next().unwrap_or("skipped").to_string()).or_default() += 1,
                }
            }
            report.push(GapRound { round: r + 1, gaps: g.len(), kept, total: self.items.len(), dropped });
            if kept == 0 {
                break;
            }
        }
        Ok(report)
    }

    /// What still looks uncovered, for tool replies.
    pub fn uncovered(&self) -> Uncovered {
        if self.items.is_empty() {
            return Uncovered { patches: 0, area: 0, top: Vec::new(), hint: "No masks yet: use find for every object, or click one object.".into() };
        }
        let g = self.gaps(0.3, 2.2);
        let area = g.iter().map(|p| p.area).sum();
        let hint = if g.is_empty() {
            "No uncovered patch looks like the masked objects.".to_string()
        } else {
            format!(
                "{} uncovered patches ({} px) have the colour of the masked objects and no mask. Check them with view (zoom on a region), click them, or call gaps with add: true.",
                g.len(),
                area
            )
        };
        Uncovered { patches: g.len(), area, top: g.into_iter().take(5).collect(), hint }
    }

    pub fn summary(&self, region: Option<Region>, ids: Option<&[u32]>) -> Vec<MaskInfo> {
        self.items
            .iter()
            .filter(|m| region.is_none_or(|r| r.overlaps(m.mask.bbox())))
            .filter(|m| ids.is_none_or(|ids| ids.contains(&m.id)))
            .map(|m| MaskInfo { id: m.id, area: m.mask.area, bbox: m.mask.bbox(), score: (m.score * 1000.0).round() / 1000.0, source: m.source.clone() })
            .collect()
    }

    pub fn selected(&self, ids: Option<&[u32]>) -> Result<Vec<&Item>> {
        if let Some(ids) = ids {
            for id in ids {
                ensure!(self.items.iter().any(|m| m.id == *id), "no mask #{id}");
            }
        }
        Ok(self.items.iter().filter(|m| ids.is_none_or(|ids| ids.contains(&m.id))).collect())
    }

    /// Numbered preview: each mask filled in its own colour with a dark outline and its id at its centre.
    /// `max_side` limits the output size; a small region is enlarged (zoom) by a whole factor.
    pub fn preview(&self, region: Option<Region>, max_side: u32, ids: Option<&[u32]>) -> Result<Preview> {
        let img = &self.image()?.rgb;
        let (w, h) = img.dimensions();
        let r = match region {
            Some(r) => r.clamp_to(w, h)?,
            None => self.full(),
        };
        let mut canvas = RgbaImage::new(r.w, r.h);
        for y in 0..r.h {
            for x in 0..r.w {
                let p = img.get_pixel(r.x + x, r.y + y).0;
                canvas.put_pixel(x, y, image::Rgba([p[0], p[1], p[2], 255]));
            }
        }
        let items = self.selected(ids)?;
        for m in &items {
            if !r.overlaps(m.mask.bbox()) {
                continue;
            }
            let c = hue(m.id);
            for (x, y) in m.mask.pixels() {
                if x < r.x || y < r.y || x >= r.x + r.w || y >= r.y + r.h {
                    continue;
                }
                let (xi, yi) = (x as i64, y as i64);
                let border = !m.mask.get_i(xi - 1, yi) || !m.mask.get_i(xi + 1, yi) || !m.mask.get_i(xi, yi - 1) || !m.mask.get_i(xi, yi + 1);
                let p = canvas.get_pixel_mut(x - r.x, y - r.y);
                if border {
                    p.0 = [20, 30, 90, 255];
                } else {
                    for (v, ck) in p.0.iter_mut().zip(c) {
                        *v = ((*v as u16 + ck as u16) / 2) as u8;
                    }
                }
            }
        }
        let side = r.w.max(r.h);
        let (out, scale) = if side > max_side {
            let s = max_side as f64 / side as f64;
            let (ow, oh) = (((r.w as f64 * s).round() as u32).max(1), ((r.h as f64 * s).round() as u32).max(1));
            (image::imageops::resize(&canvas, ow, oh, image::imageops::FilterType::Triangle), s)
        } else {
            let k = (max_side / side).max(1);
            if k == 1 {
                (canvas, 1.0)
            } else {
                (image::imageops::resize(&canvas, r.w * k, r.h * k, image::imageops::FilterType::Nearest), k as f64)
            }
        };
        let mut out = out;
        let px = if scale >= 3.0 { 2 } else { 1 };
        for m in &items {
            let (cx, cy) = m.mask.centroid();
            if cx < r.x as f64 || cy < r.y as f64 || cx >= (r.x + r.w) as f64 || cy >= (r.y + r.h) as f64 {
                continue;
            }
            let ox = ((cx - r.x as f64 + 0.5) * scale) as i64;
            let oy = ((cy - r.y as f64 + 0.5) * scale) as i64;
            draw_label(&mut out, &m.id.to_string(), ox, oy, px);
        }
        let mut png = Vec::new();
        out.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
        Ok(Preview { width: out.width(), height: out.height(), png, scale, region: r })
    }
}

/// Most tiles one find may use, and most grid points per side.
pub const MAX_TILES: usize = 64;
pub const MAX_GRID: u32 = 48;
/// Measured on a 16-core Ryzen with ViT-B: about 3.1 s per tile (4 encodes in parallel), 0.021 s per prompt.
const SECONDS_PER_TILE: f64 = 3.1;
const SECONDS_PER_PROMPT: f64 = 0.021;
/// Largest image accepted, in megapixels (the decoded RGB buffer and per-reply bitmaps scale with it).
pub const MAX_MEGAPIXELS: u64 = 100;

fn tile_count(area: Region, tile: u32) -> usize {
    let (tw, th) = (tile.min(area.w), tile.min(area.h));
    tile_starts(area.x, area.w, tw).len() * tile_starts(area.y, area.h, th).len()
}

/// Decode an image by its content (not its extension), apply EXIF orientation, refuse huge images.
pub fn read_image(path: &Path) -> Result<RgbImage> {
    let reader = image::ImageReader::open(path).with_context(|| format!("cannot read image {}", path.display()))?.with_guessed_format()?;
    let (w, h) = reader.into_dimensions().with_context(|| format!("cannot read image {}", path.display()))?;
    ensure!((w as u64) * (h as u64) <= MAX_MEGAPIXELS * 1_000_000, "{} is {w}x{h} pixels; the limit is {MAX_MEGAPIXELS} megapixels", path.display());
    let mut decoder = image::ImageReader::open(path)?.with_guessed_format()?.into_decoder().with_context(|| format!("cannot read image {}", path.display()))?;
    let orientation = image::ImageDecoder::orientation(&mut decoder).ok();
    let mut img = image::DynamicImage::from_decoder(decoder).with_context(|| format!("cannot read image {}", path.display()))?;
    if let Some(o) = orientation {
        img.apply_orientation(o);
    }
    Ok(img.to_rgb8())
}

/// Tile start positions along one axis: stride 60% of the tile, last tile flush with the end.
fn tile_starts(start: u32, len: u32, tile: u32) -> Vec<u32> {
    let stride = ((tile as f64) * 0.6).round().max(1.0) as u32;
    let mut v = Vec::new();
    let mut x = 0;
    loop {
        let s = start + x.min(len - tile);
        if !v.contains(&s) {
            v.push(s);
        }
        if x + tile >= len {
            break;
        }
        x += stride;
    }
    v
}

/// A mask touching (within 2 px) a crop edge that is inside the image was cut by the crop.
pub fn cut_by_crop(m: &Mask, r: Region, w: u32, h: u32) -> bool {
    if m.is_empty() {
        return false;
    }
    let b = m.bbox();
    let (lx0, ly0, lx1, ly1) = (b[0] - r.x, b[1] - r.y, b[2] - r.x, b[3] - r.y);
    (lx0 <= 2 && r.x > 0) || (ly0 <= 2 && r.y > 0) || (lx1 + 3 >= r.w && r.x + r.w < w) || (ly1 + 3 >= r.h && r.y + r.h < h)
}

/// Distinct colours by the golden angle.
pub fn hue(i: u32) -> [u8; 3] {
    let h = (i as f64 * 137.508) % 360.0;
    let (s, l) = (0.75f64, 0.55f64);
    let a = s * l.min(1.0 - l);
    let f = |n: f64| {
        let k = (n + h / 30.0) % 12.0;
        l - a * (-1f64).max((k - 3.0).min(9.0 - k).min(1.0))
    };
    [(f(0.0) * 255.0).round() as u8, (f(8.0) * 255.0).round() as u8, (f(4.0) * 255.0).round() as u8]
}

const DIGITS: [&str; 10] = [
    "111101101101111", "010110010010111", "111001111100111", "111001111001111", "101101111001001",
    "111100111001111", "111100111101111", "111001001001001", "111101111101111", "111101111001111",
];

/// White 3x5 digits on a black box, centred on (cx, cy).
fn draw_label(img: &mut RgbaImage, text: &str, cx: i64, cy: i64, px: i64) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let tw = text.len() as i64 * 4 * px + px;
    let th = 7 * px;
    let ox = (cx - tw / 2).clamp(0, (w - tw).max(0));
    let oy = (cy - th / 2).clamp(0, (h - th).max(0));
    let mut put = |x: i64, y: i64, c: [u8; 4]| {
        if x >= 0 && y >= 0 && x < w && y < h {
            img.put_pixel(x as u32, y as u32, image::Rgba(c));
        }
    };
    for y in 0..th {
        for x in 0..tw {
            put(ox + x, oy + y, [0, 0, 0, 255]);
        }
    }
    for (k, ch) in text.chars().enumerate() {
        let Some(d) = ch.to_digit(10) else { continue };
        let g = DIGITS[d as usize].as_bytes();
        for gy in 0..5i64 {
            for gx in 0..3i64 {
                if g[(gy * 3 + gx) as usize] == b'1' {
                    for yy in 0..px {
                        for xx in 0..px {
                            put(ox + px + (k as i64 * 4 + gx) * px + xx, oy + px + gy * px + yy, [255, 255, 255, 255]);
                        }
                    }
                }
            }
        }
    }
}
