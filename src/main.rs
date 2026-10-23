use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use maskbolt::export::{self, ExportArgs, Format};
use maskbolt::models::{self, ModelKind};
use maskbolt::{ClickArgs, FindArgs, Level, Point, Region, Session};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "maskbolt", version, about = "Segment Anything (SAM) on CPU: cut out objects, find every object, export masks and outlines.", long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Cut out the object at a point (or in a box) as a transparent PNG.
    Cutout {
        image: PathBuf,
        /// Pixel on the object, "x,y" (repeat for more).
        #[arg(long = "point", value_name = "X,Y")]
        points: Vec<String>,
        /// Pixel that is not part of the object, "x,y" (repeat for more).
        #[arg(long = "exclude", value_name = "X,Y")]
        excludes: Vec<String>,
        /// Box around the object, "x0,y0,x1,y1".
        #[arg(long = "box", value_name = "X0,Y0,X1,Y1")]
        bbox: Option<String>,
        /// best (default), smallest (a part), middle (the object) or largest (the object with what it holds).
        #[arg(long, default_value = "best")]
        level: String,
        /// Fill holes inside the mask.
        #[arg(long)]
        fill_holes: bool,
        #[arg(long)]
        out: PathBuf,
        /// Also write the outline as SVG.
        #[arg(long)]
        svg: Option<PathBuf>,
        /// Replace output files that already exist (the input image is never replaced).
        #[arg(long)]
        force: bool,
        /// vit-b (default, 375 MB) or tiny (MobileSAM, 41 MB).
        #[arg(long, default_value = "vit-b")]
        model: String,
    },
    /// Find every object with grid prompts (whole image, or overlapping tiles) and export masks.
    Find {
        image: PathBuf,
        /// Points per side of the prompt grid (default 32 for the whole image, 16 per tile).
        #[arg(long)]
        grid: Option<u32>,
        /// Tile size in pixels: overlapping crops, each re-encoded at full model resolution (finds small objects).
        #[arg(long, value_name = "PX")]
        tiles: Option<u32>,
        /// Limit to a region "x,y,w,h".
        #[arg(long)]
        region: Option<String>,
        /// Minimum predicted IoU to keep a mask (default 0.86).
        #[arg(long, default_value_t = 0.86)]
        min_score: f32,
        /// Smallest mask, as a fraction of the image area (default 0.0005).
        #[arg(long, default_value_t = 0.0005)]
        min_area: f64,
        /// Largest mask, as a fraction of the image area (default 0.25).
        #[arg(long, default_value_t = 0.25)]
        max_area: f64,
        /// After finding, click the uncovered patches that look like the found objects (two rounds).
        #[arg(long)]
        fill_gaps: bool,
        /// Fill holes in every mask.
        #[arg(long)]
        fill_holes: bool,
        /// Drop masks that are the background seen through another mask's hole (offcuts on a bed, frames).
        #[arg(long)]
        drop_hole_views: bool,
        /// JSON with outlines (holes kept), areas, boxes and scores.
        #[arg(long)]
        out: PathBuf,
        /// Numbered overlay PNG.
        #[arg(long)]
        overlay: Option<PathBuf>,
        /// SVG outlines (even-odd paths with holes).
        #[arg(long)]
        svg: Option<PathBuf>,
        /// Labels PNG: one colour per mask.
        #[arg(long)]
        labels: Option<PathBuf>,
        /// Millimetres per pixel for SVG and JSON.
        #[arg(long)]
        mm_per_px: Option<f64>,
        /// Replace output files that already exist (the input image is never replaced).
        #[arg(long)]
        force: bool,
        #[arg(long, default_value = "vit-b")]
        model: String,
    },
    /// Run the MCP server on stdio.
    Mcp {
        /// Folder the server may read and write (default: the current folder).
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long, default_value = "vit-b")]
        model: String,
    },
    /// Download and check the model weights now (they also download on first use).
    Fetch {
        #[arg(long, default_value = "vit-b")]
        model: String,
    },
    /// Print the version.
    Version,
}

fn pair(s: &str) -> Result<(f32, f32)> {
    let v: Vec<f32> = s.split(',').map(|t| t.trim().parse::<f32>()).collect::<std::result::Result<_, _>>().with_context(|| format!("expected x,y, got {s}"))?;
    if v.len() != 2 {
        bail!("expected x,y, got {s}");
    }
    Ok((v[0], v[1]))
}

fn quad(s: &str) -> Result<[f32; 4]> {
    let v: Vec<f32> = s.split(',').map(|t| t.trim().parse::<f32>()).collect::<std::result::Result<_, _>>().with_context(|| format!("expected four numbers, got {s}"))?;
    if v.len() != 4 {
        bail!("expected four comma-separated numbers, got {s}");
    }
    Ok([v[0], v[1], v[2], v[3]])
}

fn session(model: &str) -> Result<Session> {
    let kind: ModelKind = model.parse()?;
    let spec = models::spec(kind);
    if !models::is_ready(spec) && models::cache_dir().join(spec.file).is_file() {
        eprintln!("maskbolt: preparing {} from {} (checking SHA-256, converting once)", spec.name, models::cache_dir().join(spec.file).display());
    } else if !models::is_ready(spec) {
        eprintln!(
            "maskbolt: first use of {}: downloading {:.0} MB ({}, {}) into {}",
            spec.name,
            spec.bytes as f64 / 1e6,
            spec.licence,
            spec.url,
            models::cache_dir().display()
        );
    }
    let t = Instant::now();
    let mut last = 0u64;
    let (backend, fetched) = maskbolt::load_model(kind, |done, total| {
        let pct = done * 100 / total.max(1);
        if pct >= last + 10 {
            last = pct;
            eprintln!("maskbolt: {pct}% of {:.0} MB", total as f64 / 1e6);
        }
    })?;
    if fetched.downloaded {
        eprintln!("maskbolt: SHA-256 checked ({}), weights ready in {}", fetched.sha256, fetched.path);
    }
    eprintln!("maskbolt: model {} loaded in {:.2} s", backend.name(), t.elapsed().as_secs_f64());
    Ok(Session::new(backend))
}

fn write(s: &Session, format: Format, path: &Path, mm: Option<f64>, force: bool) -> Result<serde_json::Value> {
    let mut a = ExportArgs::new(format);
    a.mm_per_px = mm;
    a.overwrite = force;
    Ok(serde_json::to_value(export::export(s, &a, path)?)?)
}

/// Before any model work: refuse outputs that exist (unless --force) or that are the input image.
fn check_outputs(image: &Path, outs: &[Option<&PathBuf>], force: bool) -> Result<()> {
    let input = std::fs::canonicalize(image).ok();
    for p in outs.iter().flatten() {
        if p.exists() {
            if std::fs::canonicalize(p).ok() == input && input.is_some() {
                bail!("{} is the input image; write the output to another file", p.display());
            }
            if !force {
                bail!("{} already exists; choose a new name or pass --force to replace it", p.display());
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Version => println!("maskbolt {}", env!("CARGO_PKG_VERSION")),
        Cmd::Fetch { model } => {
            let kind: ModelKind = model.parse()?;
            let s = session(&model)?;
            let spec = models::spec(kind);
            println!("{}", json!({"model": s.backend_name(), "path": models::converted_path(spec), "source": spec.url, "sha256": spec.sha256, "licence": spec.licence}));
        }
        Cmd::Mcp { root, model } => {
            let root = match root {
                Some(r) => r,
                None => std::env::current_dir()?,
            };
            let kind: ModelKind = model.parse()?;
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            rt.block_on(maskbolt::mcp::serve_stdio(&root, kind))?;
        }
        Cmd::Cutout { image, points, excludes, bbox, level, fill_holes, out, svg, force, model } => {
            let level: Level = level.parse()?;
            let mut pts = Vec::new();
            for p in &points {
                let (x, y) = pair(p)?;
                pts.push(Point { x, y, include: true });
            }
            for p in &excludes {
                let (x, y) = pair(p)?;
                pts.push(Point { x, y, include: false });
            }
            let bbox = bbox.as_deref().map(quad).transpose()?;
            if points.is_empty() && bbox.is_none() {
                bail!("give --point x,y (on the object) or --box x0,y0,x1,y1");
            }
            check_outputs(&image, &[Some(&out), svg.as_ref()], force)?;
            let mut s = session(&model)?;
            let open = s.open(&image)?;
            let r = s.click(&ClickArgs { points: pts, bbox, level, force: true, fill_holes })?;
            let exp = write(&s, Format::Cutout, &out, None, force)?;
            let svg_out = match &svg {
                Some(p) => Some(write(&s, Format::Svg, p, None, force)?),
                None => None,
            };
            println!("{}", serde_json::to_string_pretty(&json!({"open": open, "click": r, "cutout": exp, "svg": svg_out}))?);
        }
        Cmd::Find { image, grid, tiles, region, min_score, min_area, max_area, fill_gaps, fill_holes, drop_hole_views, out, overlay, svg, labels, mm_per_px, force, model } => {
            let region = region.as_deref().map(quad).transpose()?.map(|q| Region { x: q[0] as u32, y: q[1] as u32, w: q[2] as u32, h: q[3] as u32 });
            check_outputs(&image, &[Some(&out), overlay.as_ref(), svg.as_ref(), labels.as_ref()], force)?;
            let mut s = session(&model)?;
            let open = s.open(&image)?;
            eprintln!("maskbolt: encoded {}x{} in {} ms", open.width, open.height, open.encode_ms);
            let grid = grid.unwrap_or(if tiles.is_some() { 16 } else { 32 });
            let rep = s.find(&FindArgs { grid, region, tile: tiles, min_score, min_area_frac: min_area, max_area_frac: max_area })?;
            eprintln!("maskbolt: {} masks from {} prompts on {} crops in {:.1} s", rep.total, rep.prompts, rep.crops, rep.seconds);
            let hole_views = if drop_hole_views { Some(s.remove_hole_views()) } else { None };
            let gaps = if fill_gaps { Some(s.fill_gaps(0.8, 2)?) } else { None };
            let holes = if fill_holes { Some(s.fill_holes(None)?.iter().map(|h| h.1).sum::<u64>()) } else { None };
            let mut files = vec![write(&s, Format::Json, &out, mm_per_px, force)?];
            if let Some(p) = &overlay {
                files.push(write(&s, Format::Overlay, p, None, force)?);
            }
            if let Some(p) = &svg {
                files.push(write(&s, Format::Svg, p, mm_per_px, force)?);
            }
            if let Some(p) = &labels {
                files.push(write(&s, Format::Labels, p, None, force)?);
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({"open": open, "find": rep, "removed_hole_views": hole_views, "fill_gaps": gaps, "holes_filled": holes, "masks": s.items.len(), "still_uncovered": s.uncovered(), "files": files}))?
            );
        }
    }
    Ok(())
}
