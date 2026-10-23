//! Exports: overlay PNG (numbered masks), cutout PNG (transparent), SVG (outer loops and holes, even-odd),
//! JSON (outlines, area, bbox, score) and labels PNG (one colour per mask).
use crate::outline::{outlines, Loop};
use crate::session::{hue, Item, Region, Session};
use anyhow::{bail, ensure, Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// PNG of the image with numbered, coloured masks.
    Overlay,
    /// Transparent PNG of the chosen masks, cropped to them.
    Cutout,
    /// SVG paths: outer loops and holes, fill-rule evenodd.
    Svg,
    /// JSON: outlines with holes, area, bbox, score per mask.
    Json,
    /// PNG the size of the image, one colour per mask, transparent background.
    Labels,
}

impl std::str::FromStr for Format {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "overlay" => Format::Overlay,
            "cutout" => Format::Cutout,
            "svg" => Format::Svg,
            "json" => Format::Json,
            "labels" => Format::Labels,
            _ => bail!("format must be overlay, cutout, svg, json or labels"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ExportArgs {
    pub format: Format,
    pub ids: Option<Vec<u32>>,
    /// Millimetres per pixel for SVG and JSON (default: pixels).
    pub mm_per_px: Option<f64>,
    /// Overlay only: zoom on this region.
    pub region: Option<Region>,
    /// Outline simplification tolerance in pixels (Douglas-Peucker).
    pub simplify_px: f64,
    /// Replace an existing file (never the open image).
    pub overwrite: bool,
}

impl ExportArgs {
    pub fn new(format: Format) -> Self {
        Self { format, ids: None, mm_per_px: None, region: None, simplify_px: 1.2, overwrite: false }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ExportResult {
    pub path: String,
    pub format: Format,
    pub masks: usize,
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

/// Holes and blobs smaller than this many pixels are left out of outlines.
pub const MIN_LOOP_PX: u64 = 50;

pub fn export(s: &Session, a: &ExportArgs, path: &Path) -> Result<ExportResult> {
    check_target(s, path, a.overwrite)?;
    let sel = s.selected(a.ids.as_deref())?;
    if let Some(mm) = a.mm_per_px {
        ensure!(mm > 0.0 && mm.is_finite(), "mm_per_px must be a positive number");
    }
    let (w, h) = s.size();
    let mut dims = (None, None);
    let bytes: Vec<u8> = match a.format {
        Format::Overlay => {
            let max_side = match a.region {
                Some(_) => 1200,
                None => w.max(h),
            };
            let p = s.preview(a.region, max_side, a.ids.as_deref())?;
            dims = (Some(p.width), Some(p.height));
            p.png
        }
        Format::Cutout => {
            ensure!(!sel.is_empty(), "no masks to cut out");
            let img = cutout(s, &sel)?;
            dims = (Some(img.width()), Some(img.height()));
            png(&img)?
        }
        Format::Labels => {
            let img = labels(s, &sel);
            dims = (Some(w), Some(h));
            png(&img)?
        }
        Format::Svg => svg(s, &sel, a.mm_per_px, a.simplify_px).into_bytes(),
        Format::Json => serde_json::to_vec_pretty(&json_doc(s, &sel, a.mm_per_px, a.simplify_px))?,
    };
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    std::fs::write(path, &bytes).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(ExportResult { path: path.display().to_string(), format: a.format, masks: sel.len(), bytes: bytes.len() as u64, width: dims.0, height: dims.1 })
}

/// Never replace the open image; replace other existing files only with `overwrite`.
fn check_target(s: &Session, path: &Path, overwrite: bool) -> Result<()> {
    let Ok(meta) = std::fs::symlink_metadata(path) else { return Ok(()) };
    if meta.is_dir() {
        bail!("{} is a folder, not a file", path.display());
    }
    if let Ok(img) = s.image() {
        let same = match (std::fs::canonicalize(path), std::fs::canonicalize(&img.path)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if same {
            bail!("{} is the open image; export to another file (the open image is never overwritten)", path.display());
        }
    }
    ensure!(overwrite, "{} already exists; choose a new name, or pass overwrite: true (CLI: --force) to replace it", path.display());
    Ok(())
}

fn png(img: &image::RgbaImage) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;
    Ok(out)
}

/// Transparent PNG of the union of the chosen masks, cropped to their joint box.
pub fn cutout(s: &Session, sel: &[&Item]) -> Result<image::RgbaImage> {
    let img = &s.image()?.rgb;
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for m in sel {
        let b = m.mask.bbox();
        x0 = x0.min(b[0]);
        y0 = y0.min(b[1]);
        x1 = x1.max(b[2]);
        y1 = y1.max(b[3]);
    }
    let (cw, ch) = (x1 - x0 + 1, y1 - y0 + 1);
    let mut out = image::RgbaImage::new(cw, ch);
    for m in sel {
        for (x, y) in m.mask.pixels() {
            let p = img.get_pixel(x, y).0;
            out.put_pixel(x - x0, y - y0, image::Rgba([p[0], p[1], p[2], 255]));
        }
    }
    Ok(out)
}

/// One distinct colour per mask id over the whole session (so the labels PNG and the JSON agree).
/// Ids get the preview hue when it is free; otherwise the 24-bit colour space is walked from a hash of the id
/// with an odd stride (which visits every colour once), skipping near-black. Each step is O(1), so any number
/// of masks gets colours without a long search.
pub fn label_colors(s: &Session) -> HashMap<u32, [u8; 3]> {
    const STRIDE: u32 = 0x9E_3779; // odd: the walk visits all 2^24 colours before repeating
    let dark = |c: u32| ((c >> 16) & 0xff).max((c >> 8) & 0xff).max(c & 0xff) < 48;
    let mut used: HashSet<u32> = HashSet::with_capacity(s.items.len());
    let mut out = HashMap::with_capacity(s.items.len());
    for m in &s.items {
        let h = hue(m.id);
        let mut c = (h[0] as u32) << 16 | (h[1] as u32) << 8 | h[2] as u32;
        if dark(c) || used.contains(&c) {
            c = m.id.wrapping_mul(0x9E37_79B1) >> 8;
            while dark(c) || used.contains(&c) {
                c = (c + STRIDE) & 0xff_ffff;
            }
        }
        used.insert(c);
        out.insert(m.id, [(c >> 16) as u8, (c >> 8) as u8, c as u8]);
    }
    out
}

pub fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

pub fn labels(s: &Session, sel: &[&Item]) -> image::RgbaImage {
    let (w, h) = s.size();
    let colors = label_colors(s);
    let mut out = image::RgbaImage::new(w, h);
    for m in sel {
        let c = colors[&m.id];
        for (x, y) in m.mask.pixels() {
            out.put_pixel(x, y, image::Rgba([c[0], c[1], c[2], 255]));
        }
    }
    out
}

/// Decimals that keep 0.01 px of precision at `k` output units per pixel (2 for pixels, more for small mm).
pub fn decimals(k: f64) -> usize {
    ((-(k * 0.01).log10()).ceil().max(2.0) as usize).min(9)
}

fn num(v: f64, dec: usize) -> String {
    let s = format!("{v:.dec$}");
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { &s };
    if s == "-0" { "0".into() } else { s.to_string() }
}

fn round_to(v: f64, dec: usize) -> f64 {
    let f = 10f64.powi(dec as i32);
    (v * f).round() / f
}

/// Six significant digits (areas in mm2 can be far below 0.01).
fn sig6(v: f64) -> f64 {
    if v == 0.0 {
        return 0.0;
    }
    let dec = (5 - v.abs().log10().floor() as i32).max(0) as usize;
    round_to(v, dec)
}

fn path_d(loops: &[Loop], k: f64) -> String {
    let dec = decimals(k);
    let mut d = String::new();
    for l in loops {
        if !d.is_empty() {
            d.push(' ');
        }
        for (i, p) in l.points.iter().enumerate() {
            d.push(if i == 0 { 'M' } else { 'L' });
            d.push_str(&num(p[0] * k, dec));
            d.push(',');
            d.push_str(&num(p[1] * k, dec));
        }
        d.push('Z');
    }
    d
}

pub fn svg(s: &Session, sel: &[&Item], mm: Option<f64>, eps: f64) -> String {
    let (w, h) = s.size();
    let k = mm.unwrap_or(1.0);
    let colors = label_colors(s);
    let (vw, vh) = (num(w as f64 * k, decimals(k)), num(h as f64 * k, decimals(k)));
    let (size, stroke) = match mm {
        Some(_) => (format!(r#"width="{vw}mm" height="{vh}mm""#), "0.3".to_string()),
        None => (format!(r#"width="{vw}" height="{vh}""#), "1".to_string()),
    };
    let mut out = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {vw} {vh}\" {size}>\n");
    for m in sel {
        let loops = outlines(&m.mask, eps, MIN_LOOP_PX);
        if loops.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "  <path id=\"mask-{}\" fill=\"{}\" fill-opacity=\"0.6\" fill-rule=\"evenodd\" stroke=\"#222222\" stroke-width=\"{stroke}\" d=\"{}\"/>\n",
            m.id,
            hex(colors[&m.id]),
            path_d(&loops, k)
        ));
    }
    out.push_str("</svg>\n");
    out
}

pub fn json_doc(s: &Session, sel: &[&Item], mm: Option<f64>, eps: f64) -> serde_json::Value {
    let (w, h) = s.size();
    let k = mm.unwrap_or(1.0);
    let colors = label_colors(s);
    let masks: Vec<serde_json::Value> = sel
        .iter()
        .map(|m| {
            let loops: Vec<serde_json::Value> = outlines(&m.mask, eps, MIN_LOOP_PX)
                .into_iter()
                .map(|l| {
                    json!({"hole": l.hole, "pixels": l.pixels,
                        "points": l.points.iter().map(|p| [round_to(p[0] * k, decimals(k)), round_to(p[1] * k, decimals(k))]).collect::<Vec<_>>()})
                })
                .collect();
            let mut v = json!({"id": m.id, "area_px": m.mask.area, "bbox": m.mask.bbox(), "score": (m.score * 1000.0).round() / 1000.0,
                "source": m.source, "color": hex(colors[&m.id]), "outlines": loops});
            if let Some(mm) = mm {
                v["area_mm2"] = json!(sig6((m.mask.area as f64) * mm * mm));
            }
            v
        })
        .collect();
    let image_name = s.image().ok().and_then(|i| i.path.file_name().map(|n| n.to_string_lossy().into_owned())).unwrap_or_default();
    let mut doc = json!({"image": image_name, "width": w, "height": h,
        "units": if mm.is_some() { "mm" } else { "px" }, "count": masks.len(), "masks": masks,
        "note": "Outline points are in image pixels (or mm) from the image's top-left corner; pixel (x, y) covers [x, x+1] x [y, y+1]. bbox is inclusive pixel indices."});
    if let Some(mm) = mm {
        doc["mm_per_px"] = json!(mm);
    }
    doc
}
