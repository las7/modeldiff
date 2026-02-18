use clap::{Parser, Subcommand};
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

use weight_inspect::diff;
use weight_inspect::gguf::parse_gguf;
use weight_inspect::gguf::GGUFParserError;
use weight_inspect::hash::compute_structural_hash;
use weight_inspect::safetensors::parse_safetensors;
use weight_inspect::safetensors::SafetensorsParserError;
use weight_inspect::types::Artifact;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("failed to open file '{path}': {source}")]
    FileOpen {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to read file '{path}': {source}")]
    FileRead {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse GGUF file '{path}': {source}")]
    GGUFParse {
        path: String,
        source: GGUFParserError,
    },
    #[error("failed to parse safetensors file '{path}': {source}")]
    SafetensorsParse {
        path: String,
        source: SafetensorsParserError,
    },
    #[error("JSON error: {0}")]
    Json(serde_json::Error),
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::FileRead {
            path: "unknown".to_string(),
            source: err,
        }
    }
}

#[derive(Parser)]
#[command(name = "weight-inspect")]
#[command(version = "0.1.0")]
#[command(about = "Structural identity for GGUF and safetensors model files", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Diff {
        file_a: String,
        file_b: String,
        #[arg(long)]
        json: bool,
    },
    Id {
        file: String,
        #[arg(long)]
        json: bool,
    },
    Inspect {
        file: String,
        #[arg(long)]
        json: bool,
    },
    Summary {
        file: String,
    },
}

fn detect_format(path: &Path) -> Result<Artifact, AppError> {
    let mut file = File::open(path).map_err(|e| AppError::FileOpen {
        path: path.display().to_string(),
        source: e,
    })?;
    let mut reader = BufReader::new(&file);
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|e| AppError::FileRead {
            path: path.display().to_string(),
            source: e,
        })?;

    if &magic == b"GGUF" {
        file.seek(SeekFrom::Start(0))
            .map_err(|e| AppError::FileRead {
                path: path.display().to_string(),
                source: e,
            })?;
        let mut reader = BufReader::new(file);
        return parse_gguf(&mut reader).map_err(|e| AppError::GGUFParse {
            path: path.display().to_string(),
            source: e,
        });
    }

    file.seek(SeekFrom::Start(0))
        .map_err(|e| AppError::FileRead {
            path: path.display().to_string(),
            source: e,
        })?;
    let mut reader = BufReader::new(file);
    parse_safetensors(&mut reader).map_err(|e| AppError::SafetensorsParse {
        path: path.display().to_string(),
        source: e,
    })
}

fn print_diff(result: &diff::DiffResult, json: bool) -> Result<(), AppError> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(result).map_err(AppError::Json)?
        );
        return Ok(());
    }

    println!("Structural Identity:");
    println!("  format equal: {}", result.format_equal);
    println!("  hash equal: {}", result.hash_equal);
    println!("  tensor count equal: {}", result.tensor_count_equal);
    println!("  metadata count equal: {}", result.metadata_count_equal);

    if !result.metadata_added.is_empty()
        || !result.metadata_removed.is_empty()
        || !result.metadata_changed.is_empty()
    {
        println!("\nMetadata:");
        for key in &result.metadata_added {
            println!("  + {}", key);
        }
        for key in &result.metadata_removed {
            println!("  - {}", key);
        }
        for change in &result.metadata_changed {
            println!(
                "  ~ {}: {} -> {}",
                change.key, change.old_value, change.new_value
            );
        }
    }

    if !result.tensors_added.is_empty()
        || !result.tensors_removed.is_empty()
        || !result.tensor_changes.is_empty()
    {
        println!("\nTensors:");
        for name in &result.tensors_added {
            println!("  + {}", name);
        }
        for name in &result.tensors_removed {
            println!("  - {}", name);
        }
        for change in &result.tensor_changes {
            println!("  ~ {}:", change.name);
            if let (Some(old), Some(new)) = (&change.dtype_old, &change.dtype_new) {
                println!("      dtype: {} -> {}", old, new);
            }
            if let (Some(old), Some(new)) = (&change.shape_old, &change.shape_new) {
                println!("      shape: {:?} -> {:?}", old, new);
            }
            if let (Some(old), Some(new)) = (&change.byte_length_old, &change.byte_length_new) {
                println!("      bytes: {} -> {}", old, new);
            }
        }
    }

    if !result.has_changes() {
        println!("\nNo differences found.");
    }
    Ok(())
}

fn main() -> Result<(), AppError> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Diff {
            file_a,
            file_b,
            json,
        } => {
            let artifact_a = detect_format(Path::new(&file_a))?;
            let artifact_b = detect_format(Path::new(&file_b))?;

            let hash_a = compute_structural_hash(&artifact_a);
            let hash_b = compute_structural_hash(&artifact_b);

            let mut result = diff::diff(&artifact_a, &artifact_b);
            result.hash_equal = hash_a == hash_b;

            print_diff(&result, json)?;
        }
        Commands::Id { file, json } => {
            let artifact = detect_format(Path::new(&file))?;
            let hash = compute_structural_hash(&artifact);

            if json {
                #[derive(Serialize)]
                struct IdOutput {
                    schema: u32,
                    format: String,
                    structural_hash: String,
                    tensor_count: usize,
                    metadata_count: usize,
                }
                let output = IdOutput {
                    schema: 1,
                    format: format!("{:?}", artifact.format).to_lowercase(),
                    structural_hash: hash,
                    tensor_count: artifact.tensors.len(),
                    metadata_count: artifact.metadata.len(),
                };
                println!(
                    "{}",
                    serde_json::to_string_pretty(&output).map_err(AppError::Json)?
                );
            } else {
                println!("format: {:?}", artifact.format);
                println!("structural_hash: {}", hash);
                println!("tensor_count: {}", artifact.tensors.len());
                println!("metadata_count: {}", artifact.metadata.len());
            }
        }
        Commands::Inspect { file, json } => {
            let artifact = detect_format(Path::new(&file))?;
            let hash = compute_structural_hash(&artifact);

            if json {
                #[derive(Serialize)]
                struct InspectOutput {
                    schema: u32,
                    format: String,
                    gguf_version: Option<i64>,
                    tensor_count: usize,
                    metadata_count: usize,
                    structural_hash: String,
                }
                let output = InspectOutput {
                    schema: 1,
                    format: format!("{:?}", artifact.format).to_lowercase(),
                    gguf_version: artifact.gguf_version,
                    tensor_count: artifact.tensors.len(),
                    metadata_count: artifact.metadata.len(),
                    structural_hash: hash,
                };
                println!(
                    "{}",
                    serde_json::to_string_pretty(&output).map_err(AppError::Json)?
                );
            } else {
                println!("format: {:?}", artifact.format);
                if let Some(version) = artifact.gguf_version {
                    println!("gguf_version: {}", version);
                }
                println!("tensor_count: {}", artifact.tensors.len());
                println!("metadata_count: {}", artifact.metadata.len());
                println!("structural_hash: {}", hash);

                if !artifact.tensors.is_empty() {
                    println!("\nFirst 5 tensors:");
                    for (i, (name, tensor)) in artifact.tensors.iter().take(5).enumerate() {
                        println!(
                            "  {}: {} {:?} ({})",
                            i + 1,
                            name,
                            tensor.shape,
                            tensor.dtype
                        );
                    }
                    if artifact.tensors.len() > 5 {
                        println!("  ... and {} more", artifact.tensors.len() - 5);
                    }
                }
            }
        }
        Commands::Summary { file } => {
            let artifact = detect_format(Path::new(&file))?;
            let hash = compute_structural_hash(&artifact);

            let version_str = artifact
                .gguf_version
                .map(|v| v.to_string())
                .unwrap_or_default();
            println!(
                "{},{},{},{},{}",
                format!("{:?}", artifact.format).to_lowercase(),
                version_str,
                artifact.tensors.len(),
                artifact.metadata.len(),
                hash
            );
        }
    }
    Ok(())
}
