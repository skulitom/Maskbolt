//! End-to-end tests with the real SAM ViT-B model (downloaded on first use, 375 MB). Run with
//! `cargo test --release -- --ignored`.
use maskbolt::export::{self, ExportArgs, Format};
use maskbolt::models::ModelKind;
use maskbolt::{ClickArgs, FindArgs, Level, Mask, Point, Session};
use std::path::{Path, PathBuf};

fn examples(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join(name)
}

fn session() -> Session {
    let (backend, _) = maskbolt::load_model(ModelKind::VitB, |_, _| {}).expect("model (run `maskbolt fetch` once)");
    Session::new(backend)
}

/// A prototype cutout PNG (cropped to its box, alpha = mask) placed back at its box origin.
fn proto(name: &str, x0: u32, y0: u32) -> Mask {
    let img = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(format!("proto-cutout-{name}.png"))).unwrap().to_rgba8();
    let mut m = Mask::from_fn(x0, y0, img.width(), img.height(), |x, y| img.get_pixel(x - x0, y - y0).0[3] > 0);
    m = m.tight();
    m
}

#[test]
#[ignore]
fn e2e_cutouts_match_the_prototype() {
    let mut s = session();
    let mut results = Vec::new();
    s.open(&examples("chelsea.png")).unwrap();
    let jobs = [("cat", "chelsea.png", (200.0, 150.0), Level::Best, (0, 0)), ("cup", "coffee.png", (300.0, 225.0), Level::Middle, (162, 19)), ("cup-saucer-spoon", "coffee.png", (300.0, 225.0), Level::Largest, (77, 18)), ("spoon", "coffee.png", (365.0, 290.0), Level::Best, (326, 216))];
    let mut open = "chelsea.png";
    for (name, image, (x, y), level, (bx, by)) in jobs {
        if image != open {
            s.open(&examples(image)).unwrap();
            open = image;
        }
        let r = s.click(&ClickArgs { points: vec![Point { x, y, include: true }], level, force: true, ..Default::default() }).unwrap();
        let id = r.id.unwrap();
        let m = &s.items.iter().find(|m| m.id == id).unwrap().mask;
        let iou = m.iou(&proto(name, bx, by));
        eprintln!("{name}: area {} score {:.3} IoU vs prototype {iou:.3} ({} ms)", r.area, r.score, r.ms);
        results.push((name, iou));
    }
    let good = results.iter().filter(|(_, v)| *v >= 0.9).count();
    assert!(good >= 3, "IoU >= 0.9 on {good}/4: {results:?}");
    assert!(results[0].1 >= 0.9 && results[3].1 >= 0.9, "cat and spoon: {results:?}");
}

#[test]
#[ignore]
fn e2e_bed_offcuts_match_the_true_shapes() {
    let mut s = session();
    s.open(&examples("bed.png")).unwrap();
    let rep = s.find(&FindArgs { grid: 12, min_area_frac: 0.01, max_area_frac: 0.3, ..Default::default() }).unwrap();
    let dropped = s.remove_hole_views();
    eprintln!("bed: {} masks found, {dropped:?} were the bed seen through holes, {:.1} s", rep.total, rep.seconds);
    assert_eq!(s.items.len(), 5, "{:?}", s.summary(None, None));
    let truth: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(examples("bed-truth.json")).unwrap()).unwrap();
    for piece in 1..=5 {
        let t = image::open(examples(&format!("bed-truth-{piece}.png"))).unwrap().to_luma8();
        let tm = Mask::from_fn(0, 0, t.width(), t.height(), |x, y| t.get_pixel(x, y).0[0] > 127);
        let best = s.items.iter().map(|m| m.mask.iou(&tm)).fold(0.0, f64::max);
        eprintln!("piece {piece}: IoU {best:.3}");
        assert!(best >= 0.9, "piece {piece}: IoU {best:.3}");
    }
    // SVG in mm with the holes kept: 5 paths, 6 holes in all.
    let mm = truth["mm_per_px"].as_f64().unwrap();
    let dir = std::env::temp_dir().join(format!("maskbolt-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut a = ExportArgs::new(Format::Svg);
    a.mm_per_px = Some(mm);
    export::export(&s, &a, &dir.join("bed.svg")).unwrap();
    let svg = std::fs::read_to_string(dir.join("bed.svg")).unwrap();
    assert_eq!(svg.matches("<path").count(), 5);
    let subpaths = svg.matches('M').count();
    assert_eq!(subpaths, 5 + 6, "five outer loops and six holes: {svg}");
    assert!(svg.contains(r#"width="600mm" height="400mm""#));
    std::fs::remove_dir_all(&dir).ok();
}

/// The real binary over stdio with the real model: open, click, export, and the root guard.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn e2e_mcp_stdio_with_the_real_model() {
    use rmcp::{model::CallToolRequestParams, transport::TokioChildProcess, ServiceExt};
    use serde_json::{json, Value};
    let root = std::env::temp_dir().join(format!("maskbolt-e2e-mcp-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::copy(examples("coffee.png"), root.join("coffee.png")).unwrap();
    let mut cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_maskbolt"));
    cmd.arg("mcp").arg("--root").arg(&root);
    let client = ().serve(TokioChildProcess::new(cmd).unwrap()).await.unwrap();
    let call = |name: &'static str, v: Value| {
        let client = &client;
        async move { client.call_tool(CallToolRequestParams::new(name).with_arguments(v.as_object().unwrap().clone())).await.unwrap() }
    };
    let text = |r: &rmcp::model::CallToolResult| -> Value { serde_json::from_str(&r.content.iter().find_map(|c| c.as_text()).unwrap().text).unwrap() };
    let r = call("maskbolt_open", json!({"path": "coffee.png"})).await;
    let v = text(&r);
    assert_eq!(v["opened"]["width"], 600, "{v}");
    assert_eq!(v["opened"]["model"], "sam-vit-b");
    let r = call("maskbolt_click", json!({"points": [[365, 290]]})).await;
    let v = text(&r);
    assert_eq!(v["click"]["id"], 1, "{v}");
    let area = v["click"]["area"].as_u64().unwrap();
    assert!((4500..6500).contains(&area), "spoon area {area}");
    let r = call("maskbolt_export", json!({"format": "cutout", "path": "spoon.png"})).await;
    assert_ne!(r.is_error, Some(true));
    assert!(root.join("spoon.png").is_file());
    let r = call("maskbolt_export", json!({"format": "cutout", "path": "../spoon.png"})).await;
    assert_eq!(r.is_error, Some(true));
    client.cancel().await.unwrap();
    std::fs::remove_dir_all(&root).ok();
}
