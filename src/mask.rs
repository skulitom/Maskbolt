//! A binary mask stored cropped to its bounding box (one byte per pixel inside the box), never as a full-size
//! bitmap: a dense image can have thousands of candidate masks.

/// A mask in image pixels. `x0, y0` is the top-left pixel of the box; `bits` holds `w * h` bytes (0 or 1).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Mask {
    pub x0: u32,
    pub y0: u32,
    pub w: u32,
    pub h: u32,
    pub bits: Vec<u8>,
    pub area: u64,
}

impl Mask {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.area == 0
    }

    /// Build from a predicate over the box `x0..x0+w, y0..y0+h` (absolute pixels), then shrink to the set pixels.
    pub fn from_fn(x0: u32, y0: u32, w: u32, h: u32, mut f: impl FnMut(u32, u32) -> bool) -> Self {
        let mut bits = vec![0u8; (w as usize) * (h as usize)];
        let mut area = 0;
        for y in 0..h {
            for x in 0..w {
                if f(x0 + x, y0 + y) {
                    bits[(y * w + x) as usize] = 1;
                    area += 1;
                }
            }
        }
        Self { x0, y0, w, h, bits, area }.tight()
    }

    /// Build from a full-size bitmap (`width * height` bytes, non-zero = set).
    pub fn from_full(width: u32, height: u32, full: &[u8]) -> Self {
        assert_eq!(full.len(), (width as usize) * (height as usize));
        Self { x0: 0, y0: 0, w: width, h: height, area: full.iter().filter(|&&v| v != 0).count() as u64, bits: full.iter().map(|&v| u8::from(v != 0)).collect() }.tight()
    }

    /// Shrink the box to the set pixels (an empty mask becomes `Mask::empty()`).
    pub fn tight(self) -> Self {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        let mut area = 0u64;
        for y in 0..self.h {
            let row = &self.bits[(y * self.w) as usize..((y + 1) * self.w) as usize];
            for (x, &v) in row.iter().enumerate() {
                if v != 0 {
                    let x = x as u32;
                    x0 = x0.min(x);
                    x1 = x1.max(x);
                    y0 = y0.min(y);
                    y1 = y1.max(y);
                    area += 1;
                }
            }
        }
        if area == 0 {
            return Self::empty();
        }
        if x0 == 0 && y0 == 0 && x1 + 1 == self.w && y1 + 1 == self.h {
            return Self { area, ..self };
        }
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        let mut bits = Vec::with_capacity((w * h) as usize);
        for y in y0..=y1 {
            let s = (y * self.w + x0) as usize;
            bits.extend_from_slice(&self.bits[s..s + w as usize]);
        }
        Self { x0: self.x0 + x0, y0: self.y0 + y0, w, h, bits, area }
    }

    /// Inclusive box `[x0, y0, x1, y1]` (all zeros for an empty mask).
    pub fn bbox(&self) -> [u32; 4] {
        if self.is_empty() {
            return [0, 0, 0, 0];
        }
        [self.x0, self.y0, self.x0 + self.w - 1, self.y0 + self.h - 1]
    }

    /// Is the absolute pixel (x, y) set?
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> bool {
        x >= self.x0 && y >= self.y0 && x < self.x0 + self.w && y < self.y0 + self.h && self.bits[((y - self.y0) * self.w + (x - self.x0)) as usize] != 0
    }

    #[inline]
    pub fn get_i(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && self.get(x as u32, y as u32)
    }

    /// Move the mask by (dx, dy) pixels (used to map a crop's mask into the full image).
    pub fn offset(&mut self, dx: u32, dy: u32) {
        if !self.is_empty() {
            self.x0 += dx;
            self.y0 += dy;
        }
    }

    /// Number of pixels set in both masks.
    pub fn intersection(&self, o: &Mask) -> u64 {
        if self.is_empty() || o.is_empty() {
            return 0;
        }
        let x0 = self.x0.max(o.x0);
        let y0 = self.y0.max(o.y0);
        let x1 = (self.x0 + self.w).min(o.x0 + o.w);
        let y1 = (self.y0 + self.h).min(o.y0 + o.h);
        if x0 >= x1 || y0 >= y1 {
            return 0;
        }
        let mut n = 0u64;
        for y in y0..y1 {
            let a = ((y - self.y0) * self.w + (x0 - self.x0)) as usize;
            let b = ((y - o.y0) * o.w + (x0 - o.x0)) as usize;
            let len = (x1 - x0) as usize;
            n += self.bits[a..a + len].iter().zip(&o.bits[b..b + len]).filter(|(p, q)| **p != 0 && **q != 0).count() as u64;
        }
        n
    }

    pub fn iou(&self, o: &Mask) -> f64 {
        let i = self.intersection(o);
        let u = self.area + o.area - i;
        if u == 0 { 1.0 } else { i as f64 / u as f64 }
    }

    /// Mean pixel position (x, y), in pixel-centre coordinates.
    pub fn centroid(&self) -> (f64, f64) {
        let (mut sx, mut sy) = (0f64, 0f64);
        for y in 0..self.h {
            for x in 0..self.w {
                if self.bits[(y * self.w + x) as usize] != 0 {
                    sx += (self.x0 + x) as f64;
                    sy += (self.y0 + y) as f64;
                }
            }
        }
        let a = self.area.max(1) as f64;
        (sx / a, sy / a)
    }

    /// Iterate over set pixels (absolute coordinates).
    pub fn pixels(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let w = self.w;
        self.bits.iter().enumerate().filter(|(_, &v)| v != 0).map(move |(i, _)| (self.x0 + i as u32 % w, self.y0 + i as u32 / w))
    }

    /// Fill interior holes: background pixels the outside cannot reach (4-connected) without crossing the mask.
    /// Returns the number of pixels added.
    pub fn fill_holes(&mut self) -> u64 {
        if self.is_empty() {
            return 0;
        }
        let (bw, bh) = (self.w as usize + 2, self.h as usize + 2);
        let inside = |bits: &[u8], bx: usize, by: usize| -> bool {
            bx >= 1 && by >= 1 && bx <= self.w as usize && by <= self.h as usize && bits[(by - 1) * self.w as usize + bx - 1] != 0
        };
        let mut seen = vec![false; bw * bh];
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(k) = stack.pop() {
            let (bx, by) = (k % bw, k / bw);
            let mut push = |nx: usize, ny: usize| {
                let nk = ny * bw + nx;
                if !seen[nk] && !inside(&self.bits, nx, ny) {
                    seen[nk] = true;
                    stack.push(nk);
                }
            };
            if bx + 1 < bw {
                push(bx + 1, by);
            }
            if bx > 0 {
                push(bx - 1, by);
            }
            if by + 1 < bh {
                push(bx, by + 1);
            }
            if by > 0 {
                push(bx, by - 1);
            }
        }
        let mut added = 0;
        for by in 1..bh - 1 {
            for bx in 1..bw - 1 {
                let i = (by - 1) * self.w as usize + bx - 1;
                if !seen[by * bw + bx] && self.bits[i] == 0 {
                    self.bits[i] = 1;
                    added += 1;
                }
            }
        }
        self.area += added;
        added
    }

    /// Paint the mask into a full-size bitmap of `width` columns.
    pub fn paint(&self, full: &mut [u8], width: u32, value: u8) {
        for (x, y) in self.pixels() {
            full[(y * width + x) as usize] = value;
        }
    }

    /// Bytes held by this mask (for memory reports).
    pub fn bytes(&self) -> usize {
        self.bits.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tight_and_get() {
        let m = Mask::from_fn(10, 10, 20, 20, |x, y| (15..18).contains(&x) && (12..14).contains(&y));
        assert_eq!(m.bbox(), [15, 12, 17, 13]);
        assert_eq!(m.area, 6);
        assert!(m.get(16, 13));
        assert!(!m.get(18, 13));
        assert_eq!(m.bytes(), 6);
    }

    #[test]
    fn intersection_and_iou() {
        let a = Mask::from_fn(0, 0, 10, 10, |x, _| x < 5);
        let b = Mask::from_fn(0, 0, 10, 10, |x, _| x >= 3);
        assert_eq!(a.intersection(&b), 20);
        assert!((a.iou(&b) - 20.0 / 100.0).abs() < 1e-9);
    }
}
