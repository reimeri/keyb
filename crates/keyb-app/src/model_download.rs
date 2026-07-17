use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use fs2::FileExt;
use sha2::{Digest, Sha256};

const MODEL_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true";
const MODEL_SIZE: u64 = 491_400_032;
const MODEL_SHA256: &str = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db";
const MAX_ATTEMPTS: usize = 3;

#[derive(Clone, Copy)]
struct Artifact<'a> {
    url: &'a str,
    size: u64,
    sha256: &'a str,
}

const DEFAULT_ARTIFACT: Artifact<'static> = Artifact {
    url: MODEL_URL,
    size: MODEL_SIZE,
    sha256: MODEL_SHA256,
};

pub fn ensure_default_model(path: &Path) -> Result<()> {
    ensure_model(path, DEFAULT_ARTIFACT, true, MAX_ATTEMPTS)
}

fn ensure_model(
    path: &Path,
    artifact: Artifact<'_>,
    progress: bool,
    attempts: usize,
) -> Result<()> {
    if file_has_size(path, artifact.size) {
        return Ok(());
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| anyhow!("model path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create model directory {}", parent.display()))?;

    let lock_path = suffixed_path(path, ".download.lock");
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open download lock {}", lock_path.display()))?;
    lock.lock_exclusive()
        .with_context(|| format!("lock {}", lock_path.display()))?;

    // Another process may have completed the download while this one waited.
    if file_has_size(path, artifact.size) {
        return Ok(());
    }

    let part_path = suffixed_path(path, ".part");
    normalize_partial(&part_path, artifact.size)?;
    check_disk_space(parent, &part_path, artifact.size)?;

    if progress {
        eprintln!(
            "keyb: downloading Qwen2.5-0.5B-Instruct Q4_K_M ({:.1} MiB, Apache-2.0)",
            artifact.size as f64 / 1_048_576.0
        );
        eprintln!("keyb: source: {}", artifact.url);
    }

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(30))
        .redirects(8)
        .user_agent("keyb/0.1 model-downloader")
        .build();

    let mut last_error = None;
    for attempt in 1..=attempts.max(1) {
        match download_once(&agent, &part_path, artifact, progress) {
            Ok(()) => match verify_partial(&part_path, artifact) {
                Ok(()) => {
                    publish(&part_path, path)?;
                    if progress {
                        eprintln!("keyb: model ready at {}", path.display());
                    }
                    return Ok(());
                }
                Err(error) => {
                    let _ = fs::remove_file(&part_path);
                    last_error = Some(error);
                }
            },
            Err(error) => last_error = Some(error),
        }
        if attempt < attempts {
            if progress {
                eprintln!(
                    "keyb: download interrupted; retrying ({}/{attempts})",
                    attempt + 1
                );
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("model download did not run")))
}

fn download_once(
    agent: &ureq::Agent,
    part_path: &Path,
    artifact: Artifact<'_>,
    progress: bool,
) -> Result<()> {
    let mut offset = fs::metadata(part_path).map(|m| m.len()).unwrap_or(0);
    if offset == artifact.size {
        return Ok(());
    }

    let mut request = agent.get(artifact.url).set("Accept-Encoding", "identity");
    if offset > 0 {
        request = request.set("Range", &format!("bytes={offset}-"));
    }
    let response = request
        .call()
        .map_err(|error| anyhow!("GET {}: {error}", artifact.url))?;

    let append = offset > 0 && response.status() == 206;
    if offset > 0 && !append {
        // Some servers ignore Range. Restart safely using the full response.
        offset = 0;
    }

    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(part_path)
        .with_context(|| format!("open partial model {}", part_path.display()))?;
    let mut input = response.into_reader();
    let mut downloaded = offset;
    let mut buffer = [0_u8; 64 * 1024];
    let mut last_report = Instant::now() - Duration::from_secs(2);

    loop {
        let read = input.read(&mut buffer).context("read model response")?;
        if read == 0 {
            break;
        }
        downloaded += read as u64;
        if downloaded > artifact.size {
            bail!("download exceeded expected size of {} bytes", artifact.size);
        }
        output
            .write_all(&buffer[..read])
            .context("write partial model")?;
        if progress && last_report.elapsed() >= Duration::from_secs(1) {
            report_progress(downloaded, artifact.size);
            last_report = Instant::now();
        }
    }
    output.flush().context("flush partial model")?;

    if downloaded != artifact.size {
        bail!(
            "download stopped at {downloaded} of {} bytes",
            artifact.size
        );
    }
    if progress {
        report_progress(downloaded, artifact.size);
    }
    Ok(())
}

fn verify_partial(path: &Path, artifact: Artifact<'_>) -> Result<()> {
    let actual_size = fs::metadata(path)
        .with_context(|| format!("stat partial model {}", path.display()))?
        .len();
    if actual_size != artifact.size {
        bail!(
            "model size mismatch: expected {}, got {actual_size}",
            artifact.size
        );
    }

    let mut file =
        File::open(path).with_context(|| format!("open {} for verification", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .context("read model for verification")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != artifact.sha256 {
        bail!(
            "model checksum mismatch: expected {}, got {actual_hash}",
            artifact.sha256
        );
    }
    Ok(())
}

fn publish(part_path: &Path, path: &Path) -> Result<()> {
    File::open(part_path)
        .and_then(|file| file.sync_all())
        .with_context(|| format!("sync downloaded model {}", part_path.display()))?;
    fs::rename(part_path, path)
        .with_context(|| format!("publish downloaded model at {}", path.display()))
}

fn normalize_partial(path: &Path, expected_size: u64) -> Result<()> {
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(());
    };
    if metadata.len() > expected_size {
        fs::remove_file(path)
            .with_context(|| format!("remove oversized partial model {}", path.display()))?;
    }
    Ok(())
}

fn check_disk_space(parent: &Path, part_path: &Path, expected_size: u64) -> Result<()> {
    let partial_size = fs::metadata(part_path).map(|m| m.len()).unwrap_or(0);
    let remaining = expected_size.saturating_sub(partial_size);
    let available = fs2::available_space(parent)
        .with_context(|| format!("check free space at {}", parent.display()))?;
    if available < remaining {
        bail!(
            "not enough free space at {}: need {:.1} MiB, have {:.1} MiB",
            parent.display(),
            remaining as f64 / 1_048_576.0,
            available as f64 / 1_048_576.0
        );
    }
    Ok(())
}

fn file_has_size(path: &Path, expected_size: u64) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.len() == expected_size)
        .unwrap_or(false)
}

fn suffixed_path(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    name.into()
}

fn report_progress(downloaded: u64, total: u64) {
    eprintln!(
        "keyb: model download: {:.1}/{:.1} MiB ({:.0}%)",
        downloaded as f64 / 1_048_576.0,
        total as f64 / 1_048_576.0,
        downloaded as f64 * 100.0 / total as f64
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    struct TestServer {
        url: String,
        ranges: Arc<Mutex<Vec<Option<String>>>>,
        handle: std::thread::JoinHandle<()>,
    }

    impl TestServer {
        fn serving(body: Vec<u8>, requests: usize, interrupt_first: bool) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
            let address = listener.local_addr().unwrap();
            let ranges = Arc::new(Mutex::new(Vec::new()));
            let seen_ranges = Arc::clone(&ranges);
            let handle = std::thread::spawn(move || {
                for index in 0..requests {
                    let (mut stream, _) = listener.accept().expect("accept request");
                    let range = read_range(&mut stream);
                    seen_ranges.lock().unwrap().push(range.clone());
                    if interrupt_first && index == 0 {
                        let split = body.len().min(7);
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .unwrap();
                        stream.write_all(&body[..split]).unwrap();
                        continue;
                    }

                    let offset = range
                        .as_deref()
                        .and_then(|value| value.strip_prefix("bytes="))
                        .and_then(|value| value.strip_suffix('-'))
                        .and_then(|value| value.parse::<usize>().ok())
                        .unwrap_or(0);
                    let status = if range.is_some() {
                        "206 Partial Content"
                    } else {
                        "200 OK"
                    };
                    write!(
                        stream,
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
                        body.len() - offset
                    )
                    .unwrap();
                    if range.is_some() {
                        write!(
                            stream,
                            "Content-Range: bytes {offset}-{}/{}\r\n",
                            body.len() - 1,
                            body.len()
                        )
                        .unwrap();
                    }
                    stream.write_all(b"\r\n").unwrap();
                    stream.write_all(&body[offset..]).unwrap();
                }
            });
            TestServer {
                url: format!("http://{address}/model.gguf"),
                ranges,
                handle,
            }
        }

        fn finish(self) -> Vec<Option<String>> {
            self.handle.join().expect("test server");
            Arc::try_unwrap(self.ranges).unwrap().into_inner().unwrap()
        }
    }

    fn read_range(stream: &mut TcpStream) -> Option<String> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream.read(&mut buffer).expect("read request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
        }
        String::from_utf8(request)
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix("Range: ").map(str::to_owned))
    }

    fn checksum(body: &[u8]) -> String {
        format!("{:x}", Sha256::digest(body))
    }

    #[test]
    fn downloads_verifies_and_publishes() {
        let body = b"small deterministic model".repeat(64);
        let server = TestServer::serving(body.clone(), 1, false);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let hash = checksum(&body);
        let artifact = Artifact {
            url: &server.url,
            size: body.len() as u64,
            sha256: &hash,
        };

        ensure_model(&path, artifact, false, 1).unwrap();

        assert_eq!(fs::read(&path).unwrap(), body);
        assert!(!suffixed_path(&path, ".part").exists());
        assert_eq!(server.finish(), vec![None]);
    }

    #[test]
    fn resumes_an_interrupted_download() {
        let body = b"a model large enough to split and resume".repeat(32);
        let server = TestServer::serving(body.clone(), 2, true);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let hash = checksum(&body);
        let artifact = Artifact {
            url: &server.url,
            size: body.len() as u64,
            sha256: &hash,
        };

        ensure_model(&path, artifact, false, 2).unwrap();

        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(server.finish(), vec![None, Some("bytes=7-".to_string())]);
    }

    #[test]
    fn checksum_failure_never_publishes() {
        let body = b"corrupt model".repeat(32);
        let server = TestServer::serving(body.clone(), 1, false);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let artifact = Artifact {
            url: &server.url,
            size: body.len() as u64,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        };

        let error = ensure_model(&path, artifact, false, 1).unwrap_err();

        assert!(error.to_string().contains("checksum mismatch"));
        assert!(!path.exists());
        assert!(!suffixed_path(&path, ".part").exists());
        server.finish();
    }

    #[test]
    fn complete_model_skips_network() {
        let body = b"existing model";
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        fs::write(&path, body).unwrap();
        let hash = checksum(body);
        let artifact = Artifact {
            url: "http://127.0.0.1:1/should-not-be-called",
            size: body.len() as u64,
            sha256: &hash,
        };

        ensure_model(&path, artifact, false, 1).unwrap();
        assert_eq!(fs::read(path).unwrap(), body);
    }

    #[test]
    fn concurrent_downloads_publish_once() {
        let body = b"one download for two callers".repeat(64);
        let server = TestServer::serving(body.clone(), 1, false);
        let directory = tempfile::tempdir().unwrap();
        let path = Arc::new(directory.path().join("model.gguf"));
        let hash = checksum(&body);

        let callers: Vec<_> = (0..2)
            .map(|_| {
                let path = Arc::clone(&path);
                let url = server.url.clone();
                let hash = hash.clone();
                let size = body.len() as u64;
                std::thread::spawn(move || {
                    let artifact = Artifact {
                        url: &url,
                        size,
                        sha256: &hash,
                    };
                    ensure_model(&path, artifact, false, 1)
                })
            })
            .collect();

        for caller in callers {
            caller.join().unwrap().unwrap();
        }
        assert_eq!(fs::read(path.as_ref()).unwrap(), body);
        assert_eq!(server.finish(), vec![None]);
    }
}
