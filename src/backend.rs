//! The model boundary. A backend encodes an image (or a crop of one) once, then turns prompts into SAM's three
//! nested candidate masks with predicted IoU scores. Everything else (levels, crops, filters, dedupe, gaps,
//! exports) lives outside, so it is testable with [`FakeBackend`].
use crate::mask::Mask;
use anyhow::Result;
use image::RgbImage;
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    /// true = this pixel belongs to the object, false = it does not.
    pub include: bool,
}

/// A prompt in the encoded image's own pixels (a crop's pixels when a crop was encoded).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompt {
    pub points: Vec<Point>,
    /// Optional box `[x0, y0, x1, y1]`.
    pub bbox: Option<[f32; 4]>,
}

/// SAM's answer to one prompt: three nested masks (in the encoded image's pixels) and their predicted IoU.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub masks: [Mask; 3],
    pub scores: [f32; 3],
}

/// An encoded image. `inner` is backend-specific.
pub struct Embedding {
    pub width: u32,
    pub height: u32,
    pub inner: Box<dyn Any + Send + Sync>,
}

pub trait Backend: Send + Sync {
    /// Short model name for replies, e.g. "sam-vit-b".
    fn name(&self) -> String;
    fn encode(&self, image: &RgbImage) -> Result<Embedding>;
    fn decode(&self, emb: &Embedding, prompts: &[Prompt]) -> Result<Vec<Proposal>>;
}

/// A stand-in model for tests and offline demos: flood fill from the first include point (or the box centre)
/// over pixels of similar colour. Level 0 uses a tight tolerance, level 1 a looser one, level 2 the looser one
/// with holes filled. Exclude points and the box act as barriers.
pub struct FakeBackend {
    pub tight: i32,
    pub loose: i32,
}

impl Default for FakeBackend {
    fn default() -> Self {
        Self { tight: 12, loose: 60 }
    }
}

impl Backend for FakeBackend {
    fn name(&self) -> String {
        "fake".into()
    }

    fn encode(&self, image: &RgbImage) -> Result<Embedding> {
        Ok(Embedding { width: image.width(), height: image.height(), inner: Box::new(image.clone()) })
    }

    fn decode(&self, emb: &Embedding, prompts: &[Prompt]) -> Result<Vec<Proposal>> {
        let img = emb.inner.downcast_ref::<RgbImage>().expect("fake embedding");
        Ok(prompts.iter().map(|p| self.one(img, p)).collect())
    }
}

impl FakeBackend {
    fn one(&self, img: &RgbImage, p: &Prompt) -> Proposal {
        let (w, h) = img.dimensions();
        let seed = p.points.iter().find(|q| q.include).map(|q| (q.x, q.y)).or_else(|| p.bbox.map(|b| ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)));
        let Some((sx, sy)) = seed else {
            return Proposal { masks: [Mask::empty(), Mask::empty(), Mask::empty()], scores: [0.0; 3] };
        };
        let (sx, sy) = ((sx.max(0.0) as u32).min(w - 1), (sy.max(0.0) as u32).min(h - 1));
        let barrier: Vec<(u32, u32)> = p.points.iter().filter(|q| !q.include).map(|q| (q.x.max(0.0) as u32, q.y.max(0.0) as u32)).collect();
        let bx = p.bbox.map(|b| [b[0].max(0.0) as u32, b[1].max(0.0) as u32, (b[2].max(0.0) as u32).min(w - 1), (b[3].max(0.0) as u32).min(h - 1)]);
        let fill = |tol: i32| -> Mask {
            let c0 = img.get_pixel(sx, sy).0;
            let mut full = vec![0u8; (w * h) as usize];
            let mut stack = vec![(sx, sy)];
            full[(sy * w + sx) as usize] = 1;
            let ok = |x: u32, y: u32| -> bool {
                if let Some(b) = bx {
                    if x < b[0] || y < b[1] || x > b[2] || y > b[3] {
                        return false;
                    }
                }
                if barrier.iter().any(|&(ex, ey)| ex.abs_diff(x) <= 1 && ey.abs_diff(y) <= 1) {
                    return false;
                }
                let c = img.get_pixel(x, y).0;
                (0..3).map(|i| (c[i] as i32 - c0[i] as i32).abs()).sum::<i32>() <= tol
            };
            while let Some((x, y)) = stack.pop() {
                let mut nb = Vec::with_capacity(4);
                if x > 0 {
                    nb.push((x - 1, y));
                }
                if y > 0 {
                    nb.push((x, y - 1));
                }
                if x + 1 < w {
                    nb.push((x + 1, y));
                }
                if y + 1 < h {
                    nb.push((x, y + 1));
                }
                for (nx, ny) in nb {
                    let i = (ny * w + nx) as usize;
                    if full[i] == 0 && ok(nx, ny) {
                        full[i] = 1;
                        stack.push((nx, ny));
                    }
                }
            }
            Mask::from_full(w, h, &full)
        };
        let a = fill(self.tight);
        let b = fill(self.loose);
        let mut c = b.clone();
        c.fill_holes();
        Proposal { masks: [a, b, c], scores: [0.80, 0.95, 0.90] }
    }
}
