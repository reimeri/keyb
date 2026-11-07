//! User-editable full static prompt, loaded once at startup (SPEC §9).

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use keyb_lm_llama::{DEFAULT_SYSTEM_PROMPT, MAX_SYSTEM_PROMPT_BYTES};

// Ignore empty/relative XDG directories, as required by the XDG spec. Never
// accidentally create configuration relative to the current working directory.
fn instructions_path(xdg: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    xdg.map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| {
            home.map(PathBuf::from)
                .filter(|p| p.is_absolute())
                .map(|p| p.join(".config"))
        })
        .map(|p| p.join("keyb/instructions.txt"))
}

pub fn load() -> String {
    let path = instructions_path(
        std::env::var_os("XDG_CONFIG_HOME"),
        std::env::var_os("HOME"),
    );
    load_or_default(path.as_deref())
}

fn load_or_default(path: Option<&Path>) -> String {
    let Some(path) = path else {
        eprintln!("keyb: cannot locate instructions.txt (no absolute XDG_CONFIG_HOME or HOME) — using default prompt");
        return DEFAULT_SYSTEM_PROMPT.to_owned();
    };
    match read_or_create(path) {
        Ok(prompt) => prompt,
        Err(error) => {
            eprintln!(
                "keyb: instructions file {}: {error} — using default prompt; existing file left unchanged",
                path.display()
            );
            DEFAULT_SYSTEM_PROMPT.to_owned()
        }
    }
}

fn read_or_create(path: &Path) -> io::Result<String> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            // create_new is atomic: another process (or a dangling symlink)
            // must never cause us to truncate an existing user file.
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut file) => {
                    file.write_all(DEFAULT_SYSTEM_PROMPT.as_bytes())?;
                    return Ok(DEFAULT_SYSTEM_PROMPT.to_owned());
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => File::open(path)?,
                Err(error) => return Err(error),
            }
        }
        Err(error) => return Err(error),
    };
    let mut bytes = Vec::new();
    file.take(MAX_SYSTEM_PROMPT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_SYSTEM_PROMPT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("prompt exceeds {MAX_SYSTEM_PROMPT_BYTES} bytes"),
        ));
    }
    let prompt = String::from_utf8(bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if prompt.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "prompt is empty",
        ));
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_xdg_and_home_without_using_working_directory() {
        let path = |xdg: Option<&str>, home: Option<&str>| {
            instructions_path(xdg.map(OsString::from), home.map(OsString::from))
        };
        assert_eq!(
            path(Some("/config"), Some("/home/user")),
            Some(PathBuf::from("/config/keyb/instructions.txt"))
        );
        for xdg in [None, Some(""), Some("relative")] {
            assert_eq!(
                path(xdg, Some("/home/user")),
                Some(PathBuf::from("/home/user/.config/keyb/instructions.txt"))
            );
        }
        assert_eq!(path(None, None), None);
        assert_eq!(path(Some("relative"), Some("relative")), None);
    }

    #[test]
    fn first_run_creates_default_and_missing_file_restores_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keyb/instructions.txt");
        assert_eq!(read_or_create(&path).unwrap(), DEFAULT_SYSTEM_PROMPT);
        assert_eq!(fs::read_to_string(&path).unwrap(), DEFAULT_SYSTEM_PROMPT);
        fs::remove_file(&path).unwrap();
        assert_eq!(read_or_create(&path).unwrap(), DEFAULT_SYSTEM_PROMPT);
    }

    #[test]
    fn loads_full_custom_prompt_verbatim_and_never_overwrites_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("instructions.txt");
        let prompt = "Predict Finnish text.\nKäytä arkikieltä.  \n";
        fs::write(&path, prompt).unwrap();
        for _ in 0..2 {
            assert_eq!(read_or_create(&path).unwrap(), prompt);
            assert_eq!(fs::read_to_string(&path).unwrap(), prompt);
        }
    }

    #[test]
    fn invalid_files_fall_back_without_modification() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("instructions.txt");
        for bytes in [
            vec![],
            b" \n\t".to_vec(),
            vec![0xff],
            vec![b'a'; MAX_SYSTEM_PROMPT_BYTES + 1],
        ] {
            fs::write(&path, &bytes).unwrap();
            assert_eq!(
                read_or_create(&path).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
            assert_eq!(load_or_default(Some(&path)), DEFAULT_SYSTEM_PROMPT);
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
    }

    #[test]
    fn unavailable_configuration_and_unreadable_paths_fall_back() {
        assert_eq!(load_or_default(None), DEFAULT_SYSTEM_PROMPT);
        let dir = tempfile::tempdir().unwrap();
        // A directory in place of the file is deterministically unreadable as
        // text, even when tests run as root (unlike permission-bit tests).
        assert_eq!(load_or_default(Some(dir.path())), DEFAULT_SYSTEM_PROMPT);
        let blocked = dir.path().join("blocked");
        fs::write(&blocked, "not a directory").unwrap();
        assert_eq!(
            load_or_default(Some(&blocked.join("instructions.txt"))),
            DEFAULT_SYSTEM_PROMPT
        );
        assert_eq!(fs::read_to_string(&blocked).unwrap(), "not a directory");
    }

    #[test]
    fn existing_dangling_symlink_is_not_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("instructions.txt");
        let target = dir.path().join("missing");
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert_eq!(load_or_default(Some(&path)), DEFAULT_SYSTEM_PROMPT);
        assert_eq!(fs::read_link(&path).unwrap(), target);
        assert!(!target.exists());
    }
}
