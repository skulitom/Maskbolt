//! Offline tests with the fake backend: outlines, holes, dedupe, crops, gaps, exports, the root guard.
mod common;
use common::*;
use maskbolt::export::{self, ExportArgs, Format};
use maskbolt::outline::{outlines, signed_area};
use maskbolt::paths::Root;
use maskbolt::session::cut_by_crop;
use maskbolt::{ClickArgs, FakeBackend, FindArgs, Level, Mask, Point, Region, Session};
use std::path::PathBuf;
use std::sync::Arc;

fn session(img: image::RgbImage) -> Session {
    let mut s = Session::new(Arc::new(FakeBackend::default()));
    s.open_rgb(img, PathBuf::from("synthetic.png")).unwrap();
    s
}

fn click(s: &mut Session, x: f32, y: f32) -> maskbolt::session::ClickResult {
    s.click(&ClickArgs { points: vec![Point { x, y, include: true }], ..Default::default() }).unwrap()
}

fn square_with_hole() -> Mask {
    Mask::from_fn(10, 20, 40, 40, |x, y| !((25..35).contains(&x) && (35..45).contains(&y)))
}

#[test]
fn outline_has_outer_loop_and_hole() {
    let m = square_with_hole();
    assert_eq!(m.area, 1600 - 100);
    // Unsimplified: pixel-corner coordinates, the 40x40 square spans [10, 50] x [20, 60] with corners cut by
    // half a pixel (marching squares through pixel centres).
    let raw = outlines(&m, 0.0, 1);
    assert_eq!(raw.len(), 2, "{raw:?}");
    assert!(!raw[0].hole && raw[1].hole);
    assert_eq!(signed_area(&raw[0].points), 1599.5);
    assert_eq!(signed_area(&raw[1].points), -99.5, "holes run the other way");
    // Simplified (0.5 px): a few points; the area moves by at most tolerance x perimeter / 2.
    let loops = outlines(&m, 0.5, 1);
    assert_eq!(loops.len(), 2, "{loops:?}");
    let (outer, hole) = (&loops[0], &loops[1]);
    let a = signed_area(&outer.points);
    assert!((a - 1600.0).abs() <= 0.5 * 160.0 / 2.0 + 1.0, "outer area {a}");
    let b = signed_area(&hole.points);
    assert!((b + 100.0).abs() <= 0.5 * 40.0 / 2.0 + 1.0, "hole area {b}");
    assert!(outer.points.len() <= 8, "simplified square has {} points", outer.points.len());
    let xs: Vec<f64> = outer.points.iter().map(|p| p[0]).collect();
    assert!(xs.iter().cloned().fold(f64::MAX, f64::min) >= 9.9 && xs.iter().cloned().fold(f64::MIN, f64::max) <= 50.1);
}

#[test]
fn outline_keeps_separate_blobs_and_skips_specks() {
    let m = Mask::from_fn(0, 0, 100, 40, |x, y| (x < 30 && y < 30) || ((60..90).contains(&x) && y < 30) || (x == 99 && y == 39));
    let loops = outlines(&m, 1.0, 50);
    assert_eq!(loops.iter().filter(|l| !l.hole).count(), 2, "two 900 px blobs, the 1 px speck dropped");
    // The largest blob is always kept, even below min_px.
    let tiny = Mask::from_fn(0, 0, 10, 10, |x, y| x < 3 && y < 3);
    assert_eq!(outlines(&tiny, 1.0, 50).len(), 1);
}

#[test]
fn fill_holes_fills_only_enclosed_background() {
    let mut m = square_with_hole();
    assert_eq!(m.fill_holes(), 100);
    assert_eq!(m.area, 1600);
    // A U shape has no enclosed hole.
    let mut u = Mask::from_fn(0, 0, 30, 30, |x, y| !(5..25).contains(&x) || y >= 25);
    let before = u.area;
    assert_eq!(u.fill_holes(), 0);
    assert_eq!(u.area, before);
}

#[test]
fn dedupe_rules() {
    let mut s = session(canvas(100, 100, WHITE));
    let sq = |x0: u32, y0: u32, n: u32| Mask::from_fn(x0, y0, n, n, |_, _| true);
    assert_eq!(s.add(sq(10, 10, 20), 0.9, "t", false), Ok(1));
    assert_eq!(s.add(sq(11, 11, 20), 0.9, "t", false), Err("duplicate of #1".into()));
    assert_eq!(s.add(sq(12, 12, 4), 0.9, "t", false), Err("inside #1".into()));
    assert_eq!(s.add(sq(0, 0, 45), 0.9, "t", false), Err("contains #1".into()));
    assert_eq!(s.add(sq(60, 60, 10), 0.9, "t", false), Ok(2));
    assert_eq!(s.add(sq(12, 12, 4), 0.9, "t", true), Ok(3), "force keeps it");
    assert_eq!(s.add(Mask::empty(), 0.9, "t", false), Err("empty".into()));
}

#[test]
fn crop_edge_filter() {
    let (w, h) = (300, 200);
    let tile = Region { x: 60, y: 0, w: 100, h: 100 };
    // Touches the tile's left edge (inside the image): cut.
    let m = Mask::from_fn(61, 40, 10, 10, |_, _| true);
    assert!(cut_by_crop(&m, tile, w, h));
    // Touches the tile's top edge, which is the image's top edge: not cut.
    let m = Mask::from_fn(90, 0, 10, 10, |_, _| true);
    assert!(!cut_by_crop(&m, tile, w, h));
    // Well inside: not cut.
    let m = Mask::from_fn(90, 40, 10, 10, |_, _| true);
    assert!(!cut_by_crop(&m, tile, w, h));
    // Three pixels from the right edge: cut (2-3 px margin).
    let m = Mask::from_fn(141, 40, 17, 10, |_, _| true);
    assert!(cut_by_crop(&m, tile, w, h));
}

#[test]
fn find_with_tiles_keeps_each_object_once() {
    let mut img = canvas(300, 200, WHITE);
    for (cx, cy) in [(150, 100), (40, 40), (250, 150), (90, 160)] {
        disc(&mut img, cx, cy, 18, RED);
    }
    let mut s = session(img);
    let rep = s.find(&FindArgs { grid: 8, tile: Some(100), ..Default::default() }).unwrap();
    assert_eq!(rep.crops, 5 * 3, "tiles at x 0,60,120,180,200 and y 0,60,100");
    assert_eq!(rep.total, 4, "{rep:?}");
    assert!(rep.dropped.get("cut by the crop edge").copied().unwrap_or(0) > 0, "{rep:?}");
    assert!(rep.dropped.get("crop background").copied().unwrap_or(0) > 0, "{rep:?}");
    for m in &s.items {
        assert!((m.mask.area as i64 - 1009).abs() < 60, "a disc of radius 18 has about 1009 px, got {}", m.mask.area);
    }
}

#[test]
fn find_whole_image_drops_background_as_too_large() {
    let mut img = canvas(200, 160, WHITE);
    disc(&mut img, 50, 50, 20, RED);
    disc(&mut img, 140, 100, 25, BLUE);
    let mut s = session(img);
    let rep = s.find(&FindArgs { grid: 16, ..Default::default() }).unwrap();
    assert_eq!(rep.total, 2, "{rep:?}");
    assert!(rep.dropped["too large"] > 0);
    assert!(rep.dropped["duplicate"] > 0);
}

#[test]
fn gaps_find_the_missing_look_alikes() {
    let centres = [(50, 50), (150, 50), (250, 50), (50, 200), (150, 200), (250, 200)];
    let mut img = canvas(320, 260, WHITE);
    for (x, y) in centres {
        disc(&mut img, x, y, 18, RED);
    }
    let mut s = session(img);
    for (x, y) in &centres[..3] {
        assert!(click(&mut s, *x as f32, *y as f32).id.is_some());
    }
    let g = s.gaps(0.3, 2.2);
    assert_eq!(g.len(), 3, "{g:?}");
    for p in &g {
        assert!(centres[3..].iter().any(|(x, y)| (p.x as i64 - x).pow(2) + (p.y as i64 - y).pow(2) < 18 * 18), "{p:?} is not inside a missing disc");
    }
    let u = s.uncovered();
    assert_eq!(u.patches, 3);
    assert!(u.hint.contains("3 uncovered patches"));
    let rounds = s.fill_gaps(0.8, 2).unwrap();
    assert_eq!(rounds[0].kept, 3);
    assert_eq!(s.items.len(), 6);
    assert!(s.gaps(0.3, 2.2).is_empty());
    assert_eq!(s.uncovered().patches, 0);
}

#[test]
fn click_levels_box_and_errors() {
    let mut s = session(ring_scene());
    // Best (the fake's middle mask) is the ring with its hole.
    let r = click(&mut s, 50.0, 80.0);
    assert_eq!(r.id, Some(1));
    assert_eq!(r.area, 80 * 80 - 30 * 30);
    assert_eq!(r.levels.len(), 3);
    // Largest is the ring with the hole filled.
    let r = s.click(&ClickArgs { points: vec![Point { x: 50.0, y: 80.0, include: true }], level: Level::Largest, force: true, ..Default::default() }).unwrap();
    assert_eq!(r.area, 80 * 80);
    // The same object again is a duplicate unless forced.
    let r = click(&mut s, 45.0, 45.0);
    assert!(r.skipped.as_deref().unwrap().starts_with("duplicate of #1") || r.skipped.as_deref().unwrap().starts_with("inside"), "{r:?}");
    // A box alone uses its centre.
    let r = s.click(&ClickArgs { bbox: Some([140.0, 30.0, 180.0, 70.0]), ..Default::default() }).unwrap();
    assert!(r.id.is_some());
    assert!((r.area as i64 - 709).abs() < 40, "disc r=15, got {}", r.area);
    // Fill holes on click.
    let mut s2 = session(ring_scene());
    let r = s2.click(&ClickArgs { points: vec![Point { x: 50.0, y: 80.0, include: true }], fill_holes: true, ..Default::default() }).unwrap();
    assert_eq!(r.holes_filled, Some(900));
    // Errors: outside the image, nothing to click, bad box.
    assert!(s.click(&ClickArgs { points: vec![Point { x: 500.0, y: 5.0, include: true }], ..Default::default() }).unwrap_err().to_string().contains("outside the image"));
    assert!(s.click(&ClickArgs::default()).is_err());
    assert!(s.click(&ClickArgs { bbox: Some([50.0, 50.0, 40.0, 60.0]), ..Default::default() }).is_err());
}

#[test]
fn remove_fill_holes_and_hole_views() {
    // Grey bed, a red ring; the bed seen through the ring's hole is its own grey patch.
    let mut img = canvas(200, 160, GREY);
    rect(&mut img, 40, 40, 120, 120, RED);
    rect(&mut img, 65, 65, 95, 95, GREY);
    let mut s = session(img);
    assert_eq!(click(&mut s, 50.0, 80.0).id, Some(1));
    let r = click(&mut s, 80.0, 80.0);
    assert_eq!(r.id, Some(2), "{r:?}");
    assert_eq!(r.area, 900);
    assert_eq!(s.remove_hole_views(), vec![2]);
    assert_eq!(s.items.len(), 1);
    let filled = s.fill_holes(Some(&[1])).unwrap();
    assert_eq!(filled, vec![(1, 900, 6400)]);
    assert!(s.fill_holes(Some(&[9])).is_err());
    let (removed, missing) = s.remove(&[1, 7]);
    assert_eq!((removed, missing), (vec![1], vec![7]));
    assert!(s.items.is_empty());
}

#[test]
fn preview_numbers_masks_and_zooms() {
    let mut s = session(ring_scene());
    click(&mut s, 50.0, 80.0);
    let p = s.preview(None, 1000, None).unwrap();
    assert_eq!((p.width, p.height, p.scale), (1000, 800, 5.0));
    let z = s.preview(Some(Region { x: 30, y: 30, w: 50, h: 40 }), 500, None).unwrap();
    assert_eq!((z.width, z.height), (500, 400));
    let small = s.preview(None, 100, None).unwrap();
    assert_eq!((small.width, small.height), (100, 80));
    let img = image::load_from_memory(&p.png).unwrap().to_rgba8();
    // The ring's border is drawn dark; its centre label is black and white.
    assert_eq!(img.get_pixel(40 * 5 + 2, 80 * 5).0, [20, 30, 90, 255]);
}

#[test]
fn every_export_format() {
    let dir = temp_dir("exports");
    let mut s = session(ring_scene());
    click(&mut s, 50.0, 80.0);
    click(&mut s, 160.0, 50.0);
    assert_eq!(s.items.len(), 2);
    let out = |n: &str| dir.join(n);

    let r = export::export(&s, &ExportArgs::new(Format::Overlay), &out("o.png")).unwrap();
    assert_eq!((r.width, r.height, r.masks), (Some(200), Some(160), 2));
    assert_eq!(image::open(out("o.png")).unwrap().width(), 200);

    let mut a = ExportArgs::new(Format::Cutout);
    a.ids = Some(vec![1]);
    export::export(&s, &a, &out("c.png")).unwrap();
    let c = image::open(out("c.png")).unwrap().to_rgba8();
    assert_eq!(c.dimensions(), (80, 80));
    assert_eq!(c.get_pixel(5, 5).0, [200, 30, 30, 255]);
    assert_eq!(c.get_pixel(40, 40).0[3], 0, "the hole is transparent");

    let mut a = ExportArgs::new(Format::Svg);
    a.ids = Some(vec![1]);
    a.mm_per_px = Some(0.5);
    export::export(&s, &a, &out("s.svg")).unwrap();
    let svg = std::fs::read_to_string(out("s.svg")).unwrap();
    assert!(svg.contains(r#"fill-rule="evenodd""#));
    assert!(svg.contains(r#"width="100mm" height="80mm""#), "{svg}");
    assert_eq!(svg.matches("<path").count(), 1);
    let d = svg.split(" d=\"").nth(1).unwrap();
    assert_eq!(d.matches('M').count(), 2, "outer loop and hole: {d}");
    assert!(d.starts_with("M20,") || d.contains("M20,") || d.contains(",20L") || d.contains("L20,"), "ring starts at x=40 px = 20 mm: {d}");

    let mut a = ExportArgs::new(Format::Json);
    a.mm_per_px = Some(0.5);
    export::export(&s, &a, &out("m.json")).unwrap();
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out("m.json")).unwrap()).unwrap();
    assert_eq!(j["count"], 2);
    assert_eq!(j["units"], "mm");
    let ring = &j["masks"][0];
    assert_eq!(ring["id"], 1);
    assert_eq!(ring["area_px"], 5500);
    assert_eq!(ring["area_mm2"], 1375.0);
    assert_eq!(ring["bbox"], serde_json::json!([40, 40, 119, 119]));
    let loops = ring["outlines"].as_array().unwrap();
    assert_eq!(loops.len(), 2);
    assert_eq!(loops[1]["hole"], true);
    assert!(ring["score"].as_f64().unwrap() > 0.9);

    export::export(&s, &ExportArgs::new(Format::Labels), &out("l.png")).unwrap();
    let l = image::open(out("l.png")).unwrap().to_rgba8();
    assert_eq!(l.dimensions(), (200, 160));
    assert_eq!(l.get_pixel(5, 5).0[3], 0, "background is transparent");
    assert_eq!(l.get_pixel(80, 80).0[3], 0, "the ring's hole is background");
    let (a1, a2) = (l.get_pixel(45, 45).0, l.get_pixel(160, 50).0);
    assert_eq!((a1[3], a2[3]), (255, 255));
    assert_ne!(a1, a2, "one colour per mask");
    let colours: std::collections::HashSet<[u8; 4]> = l.pixels().map(|p| p.0).filter(|p| p[3] == 255).collect();
    assert_eq!(colours.len(), 2);
    assert_eq!(ring["color"], format!("#{:02x}{:02x}{:02x}", a1[0], a1[1], a1[2]));

    assert!(export::export(&s, &{
        let mut a = ExportArgs::new(Format::Cutout);
        a.ids = Some(vec![42]);
        a
    }, &out("x.png"))
    .is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn root_guard_refuses_paths_outside() {
    let base = temp_dir("root");
    let root_dir = base.join("root");
    std::fs::create_dir_all(&root_dir).unwrap();
    std::fs::write(root_dir.join("a.png"), b"x").unwrap();
    std::fs::write(base.join("secret.png"), b"x").unwrap();
    let root = Root::new(&root_dir).unwrap();
    assert!(root.read("a.png").is_ok());
    assert!(root.read(root_dir.join("a.png").to_str().unwrap()).is_ok(), "absolute paths inside are fine");
    let e = root.read(base.join("secret.png").to_str().unwrap()).unwrap_err().to_string();
    assert!(e.contains("outside the allowed root"), "{e}");
    assert!(root.read("../secret.png").unwrap_err().to_string().contains("outside"));
    assert!(root.read("missing.png").is_err());
    let w = root.write("out/deep/m.json").unwrap();
    assert!(w.starts_with(root.dir()));
    assert!(!w.parent().unwrap().exists(), "folders are created by the export that writes, not by the check");
    assert!(root.write("../escape.png").unwrap_err().to_string().contains("outside"));
    assert!(root.write("out/../../escape.png").unwrap_err().to_string().contains("outside"));
    assert!(root.write(base.join("escape.png").to_str().unwrap()).unwrap_err().to_string().contains("outside"));
    std::fs::create_dir_all(root_dir.join("out")).unwrap();
    assert!(root.write("out").is_err(), "a folder is not a file");
    assert!(!base.join("escape.png").exists());
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn masks_are_stored_cropped() {
    let mut img = canvas(2000, 1500, WHITE);
    disc(&mut img, 1000, 700, 10, RED);
    let mut s = session(img);
    click(&mut s, 1000.0, 700.0);
    let m = &s.items[0].mask;
    assert_eq!(m.bbox(), [990, 690, 1010, 710]);
    assert!(m.bytes() <= 21 * 21, "a 21x21 box, not a 3 MB bitmap");
}

