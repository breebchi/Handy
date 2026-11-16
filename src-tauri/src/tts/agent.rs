//! Same-user, single-reply text input for the Apple Silicon prototype.

use super::TtsManager;
use hf_hub::{api::tokio::ApiBuilder, Cache, Repo, RepoType};
use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::preprocess::Lang;
use ptts::synth::{DeviceKind, Quant};
use std::fs::{self, DirBuilder};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::Shutdown;
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

const READY: &str = "ready\n";
const DONE: &str = "done\n";

fn socket_path() -> io::Result<PathBuf> {
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    let dir = PathBuf::from(format!("/tmp/handy-speech-{uid}"));
    match DirBuilder::new().mode(0o700).create(&dir) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    let metadata = fs::symlink_metadata(&dir)?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Unsafe Handy speech socket directory",
        ));
    }
    Ok(dir.join("speech.sock"))
}

pub(super) fn start(manager: Arc<TtsManager>) -> io::Result<()> {
    let path = socket_path()?;
    match fs::symlink_metadata(&path) {
        Ok(meta) if meta.file_type().is_socket() => match UnixStream::connect(&path) {
            Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => {
                fs::remove_file(&path)?
            }
            Err(error) => return Err(error),
            Ok(_) => return Err(io::ErrorKind::AddrInUse.into()),
        },
        Ok(_) => return Err(io::ErrorKind::AddrInUse.into()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let listener = UnixListener::bind(path)?;
    let (send, receive) = std::sync::mpsc::sync_channel(0);
    std::thread::spawn(move || {
        for stream in receive {
            handle_reply(&manager, stream);
        }
    });
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    if let Err(std::sync::mpsc::TrySendError::Full(mut stream)) =
                        send.try_send(stream)
                    {
                        let _ = stream.set_write_timeout(Some(Duration::from_millis(100)));
                        let _ = stream.write_all(b"Handy is already handling a speech reply.\n");
                    }
                }
                Err(error) => {
                    log::warn!("Handy speech socket accept failed: {error}");
                    break;
                }
            }
        }
    });
    Ok(())
}

fn handle_reply(manager: &TtsManager, mut stream: UnixStream) {
    if stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .is_err()
        || stream
            .set_write_timeout(Some(Duration::from_secs(1)))
            .is_err()
    {
        return;
    }
    let generation = match manager.begin_session(true) {
        Ok(generation) => generation,
        Err(error) => {
            let _ = writeln!(stream, "{error}");
            return;
        }
    };
    let result = stream
        .write_all(READY.as_bytes())
        .map_err(|error| error.to_string())
        .and_then(|()| manager.play_agent_reply(&mut stream, generation));
    let response = if manager.end_session(Some(generation), false) {
        match result {
            Ok(()) => DONE.to_string(),
            Err(error) => format!("{}\n", manager.report_failure(error)),
        }
    } else {
        "Handy stopped speech.\n".into()
    };
    let _ = stream.write_all(response.as_bytes());
}

impl TtsManager {
    fn play_agent_reply(&self, stream: &mut UnixStream, generation: u64) -> Result<(), String> {
        let input = Reader {
            stream,
            manager: self,
            generation,
        };
        self.play_phrases(input, generation, |first, session| {
            let mut engine = self
                .phonon
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if engine.is_none() {
                let directory = tauri::async_runtime::block_on(async {
                    tokio::select! {
                        result = load_checkpoint() => result.map(Some),
                        () = async {
                            while self.is_current(generation) {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                            }
                        } => Ok(None),
                    }
                })?;
                let Some(directory) = directory else {
                    return Ok(());
                };
                let checkpoint = Checkpoint::resolve(
                    directory,
                    ResolveOptions {
                        quant: Quant::Q80,
                        weights: None,
                    },
                )
                .map_err(|error| format!("Could not open Phonon: {error}"))?;
                let mut synth = checkpoint
                    .builder(Lang::En)
                    .device(DeviceKind::Cpu)
                    .build()
                    .map_err(|error| format!("Could not load Phonon: {error}"))?;
                checkpoint.register_voices(&mut synth);
                *engine = Some(synth);
            }
            let engine = engine.as_ref().ok_or("Phonon engine is unavailable.")?;
            let rate = std::num::NonZeroU32::new(engine.sample_rate())
                .ok_or("Invalid Phonon sample rate.")?;

            let mut phrase = Some(first);
            while let Some(text) = phrase {
                if !self.is_current(generation) {
                    break;
                }
                let pcm = engine
                    .stream(&text)
                    .map_err(|_| "Phonon speech synthesis failed.")?;
                for samples in pcm {
                    let samples = samples.map_err(|_| "Phonon speech synthesis failed.")?;
                    if !session.append(samples, rate, 8) {
                        return Ok(());
                    }
                }
                phrase = session.next_phrase()?;
            }
            Ok(())
        })
    }
}

async fn load_checkpoint() -> Result<PathBuf, String> {
    let api = ApiBuilder::from_cache(Cache::from_env())
        .with_token(None)
        .build()
        .map_err(|error| error.to_string())?;
    // Same immutable checkpoint as ptts 0.7.1's DEFAULT_MODEL.
    let repo = api.repo(Repo::with_revision(
        "gradium/phonon-31m-en".into(),
        RepoType::Model,
        "92741c5b07d085ff840e26fea8ce43110b87852f".into(),
    ));
    let fetch = async {
        for file in ["config.json", "tokenizer.json", "default-voice.safetensors"] {
            repo.get(file).await?;
        }
        repo.get("model.q8.gguf").await
    };
    let weights = tokio::time::timeout(Duration::from_secs(120), fetch)
        .await
        .map_err(|_| "The Phonon download timed out.".to_string())?
        .map_err(|error| format!("Could not download Phonon: {error}"))?;
    weights
        .parent()
        .map(PathBuf::from)
        .ok_or_else(|| "Phonon checkpoint is unavailable.".to_string())
}

struct Reader<'a> {
    stream: &'a mut UnixStream,
    manager: &'a TtsManager,
    generation: u64,
}

impl Read for Reader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.manager.is_current(self.generation) {
            match self.stream.read(buf) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) => {}
                result => return result,
            }
        }
        Ok(0)
    }
}

pub(crate) fn send_stdin() -> io::Result<()> {
    let stream = UnixStream::connect(socket_path()?).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("Could not reach Handy. Start the app first ({error})."),
        )
    })?;
    let mut read = BufReader::new(&stream);
    expect_line(&mut read, READY)?;
    let mut write = stream.try_clone()?;
    // The server can reject or stop a session while the producer is idle on stdin.
    let sender = std::thread::spawn(move || {
        let result = io::copy(&mut io::stdin().lock(), &mut write);
        let _ = write.shutdown(if result.is_ok() {
            Shutdown::Write
        } else {
            Shutdown::Both
        });
        result
    });
    expect_line(&mut read, DONE)?;
    sender
        .join()
        .map_err(|_| io::Error::other("Speech input thread failed"))??;
    Ok(())
}

fn expect_line(reader: &mut impl BufRead, expected: &str) -> io::Result<()> {
    let mut line = String::new();
    reader.take(4096).read_line(&mut line)?;
    if line == expected {
        return Ok(());
    }
    let message = if line.is_empty() {
        "Handy closed the speech connection."
    } else {
        line.trim_end()
    };
    Err(io::Error::other(message.to_owned()))
}
