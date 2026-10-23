#![allow(dead_code)]
use image::{Rgb, RgbImage};
use std::path::PathBuf;

pub const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
pub const RED: Rgb<u8> = Rgb([200, 30, 30]);
pub const BLUE: Rgb<u8> = Rgb([30, 60, 200]);
pub const GREY: Rgb<u8> = Rgb([90, 90, 90]);

pub fn canvas(w: u32, h: u32, c: Rgb<u8>) -> RgbImage {
    RgbImage::from_pixel(w, h, c)
}

pub fn disc(img: &mut RgbImage, cx: i64, cy: i64, r: i64, c: Rgb<u8>) {
    for y in (cy - r).max(0)..=(cy + r).min(img.height() as i64 - 1) {
        for x in (cx - r).max(0)..=(cx + r).min(img.width() as i64 - 1) {
            if (x - cx).pow(2) + (y - cy).pow(2) <= r * r {
                img.put_pixel(x as u32, y as u32, c);
            }
        }
    }
}

pub fn rect(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, c: Rgb<u8>) {
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x, y, c);
        }
    }
}

pub fn temp_dir(name: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("maskbolt-test-{name}-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A red square ring (outer 40..120, hole 65..95) and a blue disc on white, 200x160.
pub fn ring_scene() -> RgbImage {
    let mut img = canvas(200, 160, WHITE);
    rect(&mut img, 40, 40, 120, 120, RED);
    rect(&mut img, 65, 65, 95, 95, WHITE);
    disc(&mut img, 160, 50, 15, BLUE);
    img
}
