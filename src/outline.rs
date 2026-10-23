//! Outlines of a mask: every boundary loop (outer edges of each blob, and holes), traced with marching squares
//! through pixel centres and simplified with Douglas-Peucker.
//!
//! Output coordinates are image pixels with the origin at the top-left corner of the image, so pixel (x, y)
//! covers the square [x, x+1] x [y, y+1]. Outer loops run clockwise on screen (positive shoelace area with y down),
//! holes counter-clockwise, so both even-odd and non-zero fill rules draw the holes.
use crate::mask::Mask;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Loop {
    pub hole: bool,
    /// Pixels in the blob (outer loop) or in the hole.
    pub pixels: u64,
    pub points: Vec<[f64; 2]>,
}

/// All loops of a mask, outer loops first (largest first), then holes (largest first).
/// Blobs and holes smaller than `min_px` pixels are skipped, except that the largest blob is always kept.
pub fn outlines(mask: &Mask, eps: f64, min_px: u64) -> Vec<Loop> {
    if mask.is_empty() {
        return Vec::new();
    }
    // Padded local grid: one pixel of background all round, so outside background is one component.
    let (gw, gh) = (mask.w as usize + 2, mask.h as usize + 2);
    let at = |x: usize, y: usize| -> bool { x >= 1 && y >= 1 && x <= mask.w as usize && y <= mask.h as usize && mask.bits[(y - 1) * mask.w as usize + x - 1] != 0 };
    let mut label = vec![0u32; gw * gh];
    struct Comp {
        fg: bool,
        n: u64,
        bbox: [usize; 4],
        touches_border: bool,
    }
    let mut comps: Vec<Comp> = Vec::new();
    for start in 0..gw * gh {
        if label[start] != 0 {
            continue;
        }
        let fg = at(start % gw, start / gw);
        let id = comps.len() as u32 + 1;
        let mut c = Comp { fg, n: 0, bbox: [usize::MAX, usize::MAX, 0, 0], touches_border: false };
        let mut stack = vec![start];
        label[start] = id;
        while let Some(k) = stack.pop() {
            let (x, y) = (k % gw, k / gw);
            c.n += 1;
            c.bbox = [c.bbox[0].min(x), c.bbox[1].min(y), c.bbox[2].max(x), c.bbox[3].max(y)];
            if x == 0 || y == 0 || x == gw - 1 || y == gh - 1 {
                c.touches_border = true;
            }
            let mut visit = |nx: usize, ny: usize| {
                let nk = ny * gw + nx;
                if label[nk] == 0 && at(nx, ny) == fg {
                    label[nk] = id;
                    stack.push(nk);
                }
            };
            if x + 1 < gw {
                visit(x + 1, y);
            }
            if x > 0 {
                visit(x - 1, y);
            }
            if y + 1 < gh {
                visit(x, y + 1);
            }
            if y > 0 {
                visit(x, y - 1);
            }
        }
        comps.push(c);
    }
    let largest_fg = comps.iter().enumerate().filter(|(_, c)| c.fg).max_by_key(|(_, c)| c.n).map(|(i, _)| i);
    let mut out = Vec::new();
    for (i, c) in comps.iter().enumerate() {
        let keep = if c.fg { c.n >= min_px || Some(i) == largest_fg } else { !c.touches_border && c.n >= min_px };
        if !keep {
            continue;
        }
        let id = i as u32 + 1;
        let pts = trace(&label, gw, gh, id, c.bbox);
        if pts.len() < 3 {
            continue;
        }
        // Local padded grid -> image pixel corners: pixel centre of local (1,1) is image (x0+0.5, y0+0.5).
        let pts: Vec<[f64; 2]> = pts
            .into_iter()
            .map(|(x2, y2)| [mask.x0 as f64 - 1.0 + x2 as f64 / 2.0 + 0.5, mask.y0 as f64 - 1.0 + y2 as f64 / 2.0 + 0.5])
            .collect();
        let mut pts = simplify_closed(&pts, eps);
        let a = signed_area(&pts);
        let hole = !c.fg;
        // Outer loops positive (clockwise on screen), holes negative.
        if (a < 0.0) != hole {
            pts.reverse();
        }
        out.push(Loop { hole, pixels: c.n, points: pts });
    }
    out.sort_by(|a, b| a.hole.cmp(&b.hole).then(b.pixels.cmp(&a.pixels)));
    out
}

/// Marching squares around the component labelled `id`; returns its outer loop in doubled local coordinates.
fn trace(label: &[u32], gw: usize, gh: usize, id: u32, bbox: [usize; 4]) -> Vec<(i32, i32)> {
    let at = |x: i64, y: i64| -> u8 { u8::from(x >= 0 && y >= 0 && (x as usize) < gw && (y as usize) < gh && label[y as usize * gw + x as usize] == id) };
    let mut adj: HashMap<(i32, i32), Vec<(i32, i32)>> = HashMap::new();
    let mut link = |a: (i32, i32), b: (i32, i32)| {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    };
    for y in bbox[1] as i64 - 1..=bbox[3] as i64 {
        for x in bbox[0] as i64 - 1..=bbox[2] as i64 {
            let c = at(x, y) * 8 + at(x + 1, y) * 4 + at(x + 1, y + 1) * 2 + at(x, y + 1);
            let (x2, y2) = (2 * x as i32, 2 * y as i32);
            let mt = (x2 + 1, y2);
            let mr = (x2 + 2, y2 + 1);
            let mb = (x2 + 1, y2 + 2);
            let ml = (x2, y2 + 1);
            match c {
                1 | 14 => link(ml, mb),
                2 | 13 => link(mb, mr),
                3 | 12 => link(ml, mr),
                4 | 11 => link(mt, mr),
                6 | 9 => link(mt, mb),
                7 | 8 => link(ml, mt),
                5 => {
                    link(ml, mt);
                    link(mb, mr);
                }
                10 => {
                    link(mt, mr);
                    link(ml, mb);
                }
                _ => {}
            }
        }
    }
    // Walk every loop; keep the one enclosing the most area (the component's outer boundary).
    let mut best: Vec<(i32, i32)> = Vec::new();
    let mut best_area = -1.0f64;
    let mut keys: Vec<(i32, i32)> = adj.keys().copied().collect();
    keys.sort_unstable();
    for start in keys {
        if adj.get(&start).is_none_or(|n| n.is_empty()) {
            continue;
        }
        let mut lp = vec![start];
        let mut cur = adj.get_mut(&start).unwrap().pop().unwrap();
        remove_one(adj.get_mut(&cur).unwrap(), start);
        while cur != start {
            lp.push(cur);
            let Some(next) = adj.get_mut(&cur).unwrap().pop() else { break };
            remove_one(adj.get_mut(&next).unwrap(), cur);
            cur = next;
        }
        let f: Vec<[f64; 2]> = lp.iter().map(|&(x, y)| [x as f64, y as f64]).collect();
        let a = signed_area(&f).abs();
        if a > best_area {
            best_area = a;
            best = lp;
        }
    }
    best
}

fn remove_one(v: &mut Vec<(i32, i32)>, p: (i32, i32)) {
    if let Some(i) = v.iter().position(|&q| q == p) {
        v.swap_remove(i);
    }
}

/// Shoelace area (positive when the loop runs clockwise on screen, y down).
pub fn signed_area(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    let mut s = 0.0;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        s += a[0] * b[1] - b[0] * a[1];
    }
    s / 2.0
}

/// Douglas-Peucker on a closed loop: split at the point farthest from the first one, simplify both halves.
pub fn simplify_closed(lp: &[[f64; 2]], eps: f64) -> Vec<[f64; 2]> {
    if lp.len() < 4 || eps <= 0.0 {
        return lp.to_vec();
    }
    let mut far = 0;
    let mut fd = 0.0;
    for (i, p) in lp.iter().enumerate().skip(1) {
        let d = (p[0] - lp[0][0]).hypot(p[1] - lp[0][1]);
        if d > fd {
            fd = d;
            far = i;
        }
    }
    let mut first = dp(&lp[..=far], eps);
    let mut second_in: Vec<[f64; 2]> = lp[far..].to_vec();
    second_in.push(lp[0]);
    let second = dp(&second_in, eps);
    first.pop();
    first.extend_from_slice(&second[..second.len() - 1]);
    first
}

fn dp(pts: &[[f64; 2]], eps: f64) -> Vec<[f64; 2]> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let a = pts[0];
    let b = pts[pts.len() - 1];
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx.hypot(dy);
    let mut idx = 0;
    let mut dmax = 0.0;
    for (i, p) in pts.iter().enumerate().take(pts.len() - 1).skip(1) {
        let d = if len > 0.0 { (dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]).abs() / len } else { (p[0] - a[0]).hypot(p[1] - a[1]) };
        if d > dmax {
            dmax = d;
            idx = i;
        }
    }
    if dmax > eps {
        let mut l = dp(&pts[..=idx], eps);
        let r = dp(&pts[idx..], eps);
        l.pop();
        l.extend(r);
        l
    } else {
        vec![a, b]
    }
}
