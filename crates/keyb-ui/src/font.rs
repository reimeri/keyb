//! System font discovery + minimal char-to-glyph layout.
//!
//! Labels are single alphabet symbols and short ribbon strings, all simple
//! left-to-right characters, so charmap lookup + advances is enough — full
//! shaping (parley) arrives with richer text in later milestones.

use anyhow::{Context, Result};
use skrifa::instance::{LocationRef, Size};
use skrifa::MetadataProvider;
use vello::peniko::{Blob, Font};
use vello::Glyph;

pub struct FontAssets {
    pub font: Font,
    data: Vec<u8>,
}

const FALLBACK_PATHS: &[&str] = &[
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
];

impl FontAssets {
    pub fn load_system() -> Result<Self> {
        let path = fc_match_sans()
            .or_else(|| {
                FALLBACK_PATHS
                    .iter()
                    .map(std::path::PathBuf::from)
                    .find(|p| p.exists())
            })
            .context("no usable sans font found (tried fc-match and known paths)")?;
        let data =
            std::fs::read(&path).with_context(|| format!("reading font {}", path.display()))?;
        let font = Font::new(Blob::new(std::sync::Arc::new(data.clone())), 0);
        Ok(FontAssets { font, data })
    }

    fn font_ref(&self) -> Result<skrifa::FontRef<'_>> {
        skrifa::FontRef::from_index(&self.data, 0).context("parsing font")
    }

    /// Whether the font has a real glyph for `ch` (not notdef). Lets callers
    /// pick an ASCII stand-in for symbols the system font may lack.
    pub fn has_glyph(&self, ch: char) -> bool {
        self.font_ref()
            .map(|fref| fref.charmap().map(ch).is_some())
            .unwrap_or(false)
    }

    /// Lay out `text` at `size` px: glyphs with pen positions plus total advance.
    pub fn layout(&self, text: &str, size: f32) -> (Vec<Glyph>, f32) {
        let Ok(fref) = self.font_ref() else {
            return (Vec::new(), 0.0);
        };
        let charmap = fref.charmap();
        let metrics = fref.glyph_metrics(Size::new(size), LocationRef::default());
        let mut pen = 0.0f32;
        let mut glyphs = Vec::with_capacity(text.chars().count());
        for ch in text.chars() {
            // '_' is a safe stand-in if the open-box space glyph is missing; a
            // low bar stays clearly distinct from a period (unlike a mid-dot).
            let gid = charmap
                .map(ch)
                .or_else(|| if ch == '␣' { charmap.map('_') } else { None })
                .unwrap_or_default();
            glyphs.push(Glyph {
                id: gid.to_u32(),
                x: pen,
                y: 0.0,
            });
            pen += metrics.advance_width(gid).unwrap_or(size * 0.5);
        }
        (glyphs, pen)
    }
}

fn fc_match_sans() -> Option<std::path::PathBuf> {
    let out = std::process::Command::new("fc-match")
        .args(["-f", "%{file}", "sans"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = std::path::PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    path.exists().then_some(path)
}
