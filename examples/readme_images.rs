//! Regenerates every image the README shows into `data/examples/`.
//!
//! Each image is one run with fixed flags and seed, so the set rebuilds exactly:
//! `cargo run --release --example readme_images`.

use greedytile::io::cli::{Cli, FileProcessor, OutputOptions, TileOptions};
use image::codecs::gif::{GifDecoder, GifEncoder, Repeat};
use image::imageops::{self, FilterType};
use image::{AnimationDecoder, Frame};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Every README image is shown at four times its generated size
const SCALE: u32 = 4;

/// One generation run: canvas, iteration cap, seed and flags
#[derive(Clone, Copy)]
struct Run {
    size: Option<(usize, usize)>,
    iterations: usize,
    seed: u64,
    rotate: bool,
    mirror: bool,
    prefill: bool,
}

impl Run {
    /// The CLI defaults: unbounded, 1000 iterations, seed 42, no flags
    const DEFAULT: Self = Self {
        size: None,
        iterations: 1000,
        seed: 42,
        rotate: false,
        mirror: false,
        prefill: false,
    };

    const fn canvas(self, width: usize, height: usize) -> Self {
        Self {
            size: Some((width, height)),
            ..self
        }
    }

    const fn iterations(self, iterations: usize) -> Self {
        Self { iterations, ..self }
    }

    const fn seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }

    const fn rotate(self) -> Self {
        Self {
            rotate: true,
            ..self
        }
    }

    const fn mirror(self) -> Self {
        Self {
            mirror: true,
            ..self
        }
    }

    const fn prefill(self) -> Self {
        Self {
            prefill: true,
            ..self
        }
    }
}

/// The Examples table: a 128x32 result and a separate 32x32 replay per source
const EXAMPLES: [(&str, Run, Run); 7] = [
    (
        "a",
        wide().rotate().mirror().seed(9),
        replay().rotate().mirror().seed(2),
    ),
    ("b", wide().rotate().mirror(), replay().rotate().mirror()),
    (
        "c",
        wide().rotate().mirror().seed(11),
        replay().rotate().mirror(),
    ),
    ("d", wide(), replay()),
    (
        "e",
        wide().rotate().mirror().seed(10),
        replay().rotate().mirror().seed(10),
    ),
    ("f", wide(), replay()),
    ("g", wide().rotate().mirror(), replay().rotate().mirror()),
];

/// The Options table: each pair differs only in the flag it shows
const OPTIONS: [(&str, &str, Run); 12] = [
    (
        "seed_default",
        "a",
        Run::DEFAULT
            .canvas(28, 28)
            .iterations(6000)
            .rotate()
            .mirror(),
    ),
    (
        "seed_custom",
        "a",
        Run::DEFAULT
            .canvas(28, 28)
            .iterations(6000)
            .rotate()
            .mirror()
            .seed(2),
    ),
    ("iterations_default", "f", Run::DEFAULT.canvas(24, 24)),
    (
        "iterations_custom",
        "f",
        Run::DEFAULT.canvas(24, 24).iterations(200),
    ),
    ("size_unbounded", "f", Run::DEFAULT),
    ("size_24x24", "f", Run::DEFAULT.canvas(24, 24)),
    ("prefill_off", "c", small(6000).rotate().mirror().seed(2)),
    (
        "prefill_on",
        "c",
        small(6000).rotate().mirror().seed(2).prefill(),
    ),
    ("rotate_off", "g", small(3000).seed(3)),
    ("rotate_on", "g", small(3000).seed(3).rotate()),
    ("mirror_off", "chiral", small(3000).seed(1)),
    ("mirror_on", "chiral", small(3000).seed(1).mirror()),
];

const fn wide() -> Run {
    Run::DEFAULT.canvas(128, 32).iterations(30_000)
}

const fn replay() -> Run {
    Run::DEFAULT.canvas(32, 32).iterations(10_000)
}

const fn small(iterations: usize) -> Run {
    Run::DEFAULT.canvas(32, 32).iterations(iterations)
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data = root.join("data");
    let inputs = data.join("examples").join("inputs");
    let out = data.join("examples");
    let work = tempfile::tempdir()?;

    let source = |name: &str| -> PathBuf {
        let in_data = data.join(format!("{name}.png"));
        if in_data.exists() {
            in_data
        } else {
            inputs.join(format!("{name}.png"))
        }
    };

    let generate =
        |tag: &str, name: &str, run: Run, visualize: bool| -> Result<PathBuf, Box<dyn Error>> {
            let input = work.path().join(format!("{tag}.png"));
            std::fs::copy(source(name), &input)?;
            if run.prefill {
                std::fs::copy(
                    inputs.join(format!("{name}_pre.png")),
                    work.path().join(format!("{tag}_pre.png")),
                )?;
            }
            FileProcessor::new(Cli {
                target: input,
                seed: run.seed,
                iterations: run.iterations,
                quiet: true,
                no_skip: true,
                width: run.size.map(|(width, _)| width),
                height: run.size.map(|(_, height)| height),
                tiles: TileOptions {
                    prefill: run.prefill,
                    rotate: run.rotate,
                    mirror: run.mirror,
                },
                outputs: OutputOptions {
                    visualize,
                    analysis: false,
                },
            })
            .process()?;
            Ok(work.path().join(tag))
        };

    for (name, result, replay) in EXAMPLES {
        upscale_png(&source(name), &out.join(format!("{name}_4x.png")))?;
        let wide_run = generate(&format!("{name}_wide"), name, result, false)?;
        upscale_png(
            &with_suffix(&wide_run, "_result.png"),
            &out.join(format!("{name}_result_4x.png")),
        )?;
        let replay_run = generate(&format!("{name}_replay"), name, replay, true)?;
        upscale_gif(
            &with_suffix(&replay_run, "_visualization.gif"),
            &out.join(format!("{name}_visualization_4x.gif")),
        )?;
    }

    for (output, name, run) in OPTIONS {
        let option_run = generate(output, name, run, false)?;
        upscale_png(
            &with_suffix(&option_run, "_result.png"),
            &out.join(format!("{output}_4x.png")),
        )?;
    }

    Ok(())
}

fn with_suffix(stem: &Path, suffix: &str) -> PathBuf {
    let mut path = stem.as_os_str().to_owned();
    path.push(suffix);
    PathBuf::from(path)
}

fn upscale_png(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let image = image::open(input)?.to_rgba8();
    let (width, height) = image.dimensions();
    imageops::resize(&image, width * SCALE, height * SCALE, FilterType::Nearest).save(output)?;
    Ok(())
}

fn upscale_gif(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let frames = GifDecoder::new(BufReader::new(File::open(input)?))?
        .into_frames()
        .collect_frames()?;
    let mut encoder = GifEncoder::new(File::create(output)?);
    encoder.set_repeat(Repeat::Infinite)?;
    encoder.encode_frames(frames.into_iter().map(|frame| {
        let (left, top, delay) = (frame.left(), frame.top(), frame.delay());
        let buffer = frame.into_buffer();
        let (width, height) = buffer.dimensions();
        let scaled = imageops::resize(&buffer, width * SCALE, height * SCALE, FilterType::Nearest);
        Frame::from_parts(scaled, left * SCALE, top * SCALE, delay)
    }))?;
    Ok(())
}
