//! The MCP server's root guard: every read and write stays inside one directory.
//!
//! A requested path is first resolved lexically against the root (no file-system call): UNC and device paths
//! (`\\server\share`, `\\?\`, `\\.\`, `//host`), other drives, drive-relative paths, Windows names that the
//! system would silently rewrite (trailing dots or spaces, `:` streams) and anything that `..` takes out of the
//! root are refused with one message, whether or not the file exists. Only then is the file system consulted,
//! to resolve links and junctions, and the result is checked again.
use anyhow::{bail, Context, Result};
use std::path::{Component, Path, PathBuf, Prefix};

#[derive(Clone, Debug)]
pub struct Root {
    dir: PathBuf,
}

impl Root {
    pub fn new(dir: &Path) -> Result<Self> {
        let dir = canonical(dir).with_context(|| format!("root directory {} does not exist", dir.display()))?;
        if !dir.is_dir() {
            bail!("root {} is not a directory", dir.display());
        }
        Ok(Self { dir })
    }

    /// A root for the MCP server: like `new`, but a drive or file-system root and the home folder are refused,
    /// since they would expose (and let exports overwrite) far more than one project.
    pub fn for_server(dir: &Path) -> Result<Self> {
        let r = Self::new(dir)?;
        let why = if r.dir.parent().is_none() {
            Some("a drive or file-system root")
        } else if home().and_then(|h| canonical(&h).ok()).is_some_and(|h| same_path(&h, &r.dir)) {
            Some("your home folder")
        } else {
            None
        };
        if let Some(why) = why {
            bail!(
                "refusing to serve {} ({why}): every file below it would be readable and writable by the agent. \
                 Start the server in a project or image folder, or pass --root <folder>.",
                r.dir.display()
            );
        }
        Ok(r)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn outside(&self, p: &str) -> anyhow::Error {
        anyhow::anyhow!("{p} is outside the allowed root {}: use a path inside it (relative paths are resolved against the root)", self.dir.display())
    }

    /// Resolve `p` against the root without touching the file system; refuse anything not inside it.
    pub fn lexical(&self, p: &str) -> Result<PathBuf> {
        let raw = p.trim();
        if raw.is_empty() {
            bail!("empty path");
        }
        let b = raw.as_bytes();
        let sep = |c: u8| c == b'\\' || c == b'/';
        if b.len() >= 2 && sep(b[0]) && sep(b[1]) {
            // UNC (\\server\share, //host), verbatim (\\?\) and device (\\.\) paths.
            return Err(self.outside(p));
        }
        let mut comps = Path::new(raw).components().peekable();
        let mut out = match comps.peek() {
            Some(Component::Prefix(px)) => {
                let Prefix::Disk(d) = px.kind() else { return Err(self.outside(p)) };
                if root_drive(&self.dir).map(|r| r.to_ascii_uppercase()) != Some(d.to_ascii_uppercase()) {
                    return Err(self.outside(p));
                }
                comps.next();
                // "C:x" is relative to the drive's current folder: refuse it.
                if !matches!(comps.next(), Some(Component::RootDir)) {
                    return Err(self.outside(p));
                }
                PathBuf::from(format!("{}:\\", d.to_ascii_uppercase() as char))
            }
            Some(Component::RootDir) => {
                // On Windows "\x" means the current drive's root: refuse it rather than guess.
                if cfg!(windows) {
                    return Err(self.outside(p));
                }
                comps.next();
                PathBuf::from("/")
            }
            _ => self.dir.clone(),
        };
        for c in comps {
            match c {
                Component::CurDir => {}
                Component::ParentDir => {
                    if !out.pop() {
                        return Err(self.outside(p));
                    }
                }
                Component::Normal(n) => {
                    let s = n.to_string_lossy();
                    if cfg!(windows) && (s.ends_with('.') || s.ends_with(' ') || s.contains(':')) {
                        bail!("{p}: file and folder names may not end with a dot or a space or contain ':'");
                    }
                    out.push(n);
                }
                _ => return Err(self.outside(p)),
            }
        }
        if !inside(&out, &self.dir) {
            return Err(self.outside(p));
        }
        Ok(out)
    }

    /// An existing file inside the root (links and junctions resolved and checked).
    pub fn read(&self, p: &str) -> Result<PathBuf> {
        let full = self.lexical(p)?;
        let c = canonical(&full).map_err(|_| anyhow::anyhow!("{p} was not found inside the allowed root {}", self.dir.display()))?;
        if !inside(&c, &self.dir) {
            return Err(self.outside(p));
        }
        if !c.is_file() {
            bail!("{p} is not a file");
        }
        Ok(c)
    }

    /// A file to write inside the root. Nothing is created here; existing links are refused; folders that exist
    /// on the way are resolved (junctions) and checked.
    pub fn write(&self, p: &str) -> Result<PathBuf> {
        let full = self.lexical(p)?;
        if full == self.dir {
            bail!("{p} is a folder, not a file");
        }
        let mut existing = full.parent().context("no parent folder")?.to_path_buf();
        while std::fs::symlink_metadata(&existing).is_err() {
            existing = existing.parent().context("no existing parent folder")?.to_path_buf();
        }
        let base = canonical(&existing)?;
        if !inside(&base, &self.dir) {
            return Err(self.outside(p));
        }
        if let Ok(meta) = std::fs::symlink_metadata(&full) {
            if meta.file_type().is_symlink() {
                bail!("{p} is a link; exports do not write through links");
            }
            if meta.is_dir() {
                bail!("{p} is a folder, not a file");
            }
            if !inside(&canonical(&full)?, &self.dir) {
                return Err(self.outside(p));
            }
        }
        Ok(full)
    }
}

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).filter(|h| !h.is_empty()).map(PathBuf::from)
}

fn root_drive(p: &Path) -> Option<u8> {
    match p.components().next() {
        Some(Component::Prefix(px)) => match px.kind() {
            Prefix::Disk(d) | Prefix::VerbatimDisk(d) => Some(d),
            _ => None,
        },
        _ => None,
    }
}

fn same_path(a: &Path, b: &Path) -> bool {
    inside(a, b) && inside(b, a)
}

/// Component-wise prefix test (case-insensitive on Windows).
fn inside(p: &Path, root: &Path) -> bool {
    let norm = |c: Component| {
        let s = c.as_os_str().to_string_lossy().into_owned();
        if cfg!(windows) { s.to_lowercase() } else { s }
    };
    let mut pc = p.components();
    for rc in root.components() {
        match pc.next() {
            Some(c) if norm(c) == norm(rc) => {}
            _ => return false,
        }
    }
    true
}

/// `fs::canonicalize` without Windows' verbatim prefix on drive paths.
pub fn canonical(path: &Path) -> std::io::Result<PathBuf> {
    let full = std::fs::canonicalize(path)?;
    let text = full.to_string_lossy();
    Ok(match text.strip_prefix(r"\\?\") {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => PathBuf::from(rest),
        _ => full.clone(),
    })
}
