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
#[cfg(feature = "onnx")]
use weight_inspect::onnx::parse_onnx;
#[cfg(feature = "onnx")]
use weight_inspect::onnx::OnnxParserError;
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
    #[cfg(feature = "onnx")]
    #[error("failed to parse ONNX file '{path}': {source}")]
    OnnxParse {
        path: String,
        source: OnnxParserError,
    },
    #[error("ONNX support not enabled: rebuild with --features onnx")]
    OnnxNotSupported { path: String },
    #[error("invalid format '{format}': must be 'text' or 'md'")]
    InvalidFormat { format: String },
    #[error("JSON error: {0}")]
    Json(serde_json::Error),
}

impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        AppError::Json(err)
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
    /// Compare two model files and show structural differences
    Diff {
        file_a: String,
        file_b: String,
        #[arg(long, default_value = "false")]
        json: bool,
        #[arg(long, default_value = "text")]
        format: String,
        #[arg(long, default_value = "false")]
        fail_on_diff: bool,
        #[arg(long, default_value = "false")]
        only_changes: bool,
        #[arg(long, default_value = "false")]
        verbose: bool,
    },
    /// Show stable structural fingerprint
    Id {
        file: String,
        #[arg(long, default_value = "false")]
        json: bool,
    },
    /// Show full model structure details
    Inspect {
        file: String,
        #[arg(long, default_value = "false")]
        json: bool,
        #[arg(long, default_value = "false")]
        html: bool,
        #[arg(long, default_value = "false")]
        verbose: bool,
    },
    /// One-line summary for scripts and CI
    Summary { file: String },
}

fn detect_format(path: &Path) -> Result<Artifact, AppError> {
    // Check for .onnx extension first (before magic byte detection)
    if path.extension().is_some_and(|e| e == "onnx") {
        #[cfg(feature = "onnx")]
        {
            let file = File::open(path).map_err(|e| AppError::FileOpen {
                path: path.display().to_string(),
                source: e,
            })?;
            let mut reader = BufReader::new(file);
            return parse_onnx(&mut reader).map_err(|e| AppError::OnnxParse {
                path: path.display().to_string(),
                source: e,
            });
        }
        #[cfg(not(feature = "onnx"))]
        {
            return Err(AppError::OnnxNotSupported {
                path: path.display().to_string(),
            });
        }
    }

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
            format,
            fail_on_diff,
            only_changes,
            verbose,
        } => {
            let artifact_a = detect_format(Path::new(&file_a))?;
            let artifact_b = detect_format(Path::new(&file_b))?;

            let hash_a = compute_structural_hash(&artifact_a)?;
            let hash_b = compute_structural_hash(&artifact_b)?;

            let mut result = diff::diff(&artifact_a, &artifact_b);
            result.hash_equal = hash_a == hash_b;

            if fail_on_diff && result.has_changes() {
                std::process::exit(1);
            }

            if format != "text" && format != "md" {
                return Err(AppError::InvalidFormat {
                    format: format.clone(),
                });
            }

            print_diff_extended(&result, json, &format, only_changes, verbose)?;
        }
        Commands::Id { file, json } => {
            let artifact = detect_format(Path::new(&file))?;
            let hash = compute_structural_hash(&artifact)?;

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
                println!("Structural identity");
                println!("──────────────────");
                println!(
                    "ID:        wi:{}:{}:{}",
                    format!("{:?}", artifact.format).to_lowercase(),
                    artifact.gguf_version.unwrap_or(0),
                    &hash[..8.min(hash.len())]
                );
                println!("Stable:    yes (machine independent)");
                println!("Includes:  header, tensor names, shapes, dtypes");
                println!("Excludes:  raw weight bytes");
                println!("\nformat: {:?}", artifact.format);
                println!("structural_hash: {}", hash);
                println!("tensor_count: {}", artifact.tensors.len());
                println!("metadata_count: {}", artifact.metadata.len());
            }
        }
        Commands::Inspect {
            file,
            json,
            html,
            verbose,
        } => {
            let artifact = detect_format(Path::new(&file))?;
            let hash = compute_structural_hash(&artifact)?;

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
            } else if html {
                print_inspect_html(&artifact, &hash);
            } else {
                print_inspect(&artifact, &hash, verbose);
            }
        }
        Commands::Summary { file } => {
            let artifact = detect_format(Path::new(&file))?;
            let hash = compute_structural_hash(&artifact)?;

            let version_str = artifact
                .gguf_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "N/A".to_string());
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

fn print_inspect(artifact: &Artifact, hash: &str, verbose: bool) {
    let format_str = format!("{:?}", artifact.format).to_lowercase();
    let version_str = artifact
        .gguf_version
        .map(|v| v.to_string())
        .unwrap_or_else(|| "N/A".to_string());

    println!("{}", format_str);
    println!("───");
    println!("Version:  {}", version_str);
    println!("Tensors:  {}", artifact.tensors.len());
    println!("Metadata: {}", artifact.metadata.len());

    if verbose {
        println!("\nStructural summary");
        println!("──────────────────");

        // Count dtypes
        let mut dtype_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for tensor in artifact.tensors.values() {
            *dtype_counts.entry(&tensor.dtype).or_insert(0) += 1;
        }

        let total: usize = dtype_counts.values().sum();
        println!(
            "Dtypes:  {}",
            if total > 0 {
                dtype_counts
                    .iter()
                    .map(|(k, v)| format!("{} ({:.1}%)", k, (*v as f64) * 100.0 / total as f64))
                    .collect::<Vec<_>>()
                    .join(", ")
            } else {
                "N/A".to_string()
            }
        );
    }

    println!("\nStructural ID");
    println!("─────────────");
    println!("Hash: {}", hash);

    if !artifact.tensors.is_empty() && verbose {
        println!("\nTensors (first 10)");
        println!("──────────────────");
        println!("{:<40} {:<8} {:<20} Bytes", "Name", "Dtype", "Shape");
        println!("{:<40} {:<8} {:<20} -----", "", "", "");
        for (name, tensor) in artifact.tensors.iter().take(10) {
            let shape_str = format!("{:?}", tensor.shape);
            let display_name = if name.len() > 40 {
                format!("{}...", &name[..37])
            } else {
                name.clone()
            };
            println!(
                "{:<40} {:<8} {:<20} {}",
                display_name, tensor.dtype, shape_str, tensor.byte_length
            );
        }
    }
}

fn print_inspect_html(artifact: &Artifact, hash: &str) {
    println!("<!DOCTYPE html>");
    println!("<html><head><title>Model Inspection</title>");
    println!("<style>");
    println!("body {{ font-family: system-ui, sans-serif; margin: 2rem; }}");
    println!("table {{ border-collapse: collapse; width: 100%; }}");
    println!("th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}");
    println!("th {{ background: #f5f5f5; }}");
    println!(".summary {{ background: #f9f9f9; padding: 1rem; margin-bottom: 1rem; }}");
    println!("</style></head><body>");
    println!("<h1>Model Inspection</h1>");
    println!("<div class='summary'>");
    println!("<p><strong>Format:</strong> {:?}</p>", artifact.format);
    if let Some(v) = artifact.gguf_version {
        println!("<p><strong>Version:</strong> {}</p>", v);
    }
    println!(
        "<p><strong>Tensors:</strong> {}</p>",
        artifact.tensors.len()
    );
    println!(
        "<p><strong>Metadata:</strong> {}</p>",
        artifact.metadata.len()
    );
    println!("<p><strong>Hash:</strong> {}</p>", hash);
    println!("</div>");
    println!("<h2>Tensors</h2>");
    println!("<table><tr><th>Name</th><th>Dtype</th><th>Shape</th><th>Bytes</th></tr>");
    for (name, tensor) in artifact.tensors.iter().take(100) {
        println!(
            "<tr><td>{}</td><td>{}</td><td>{:?}</td><td>{}</td></tr>",
            name, tensor.dtype, tensor.shape, tensor.byte_length
        );
    }
    if artifact.tensors.len() > 100 {
        println!(
            "<tr><td colspan='4'>... and {} more</td></tr>",
            artifact.tensors.len() - 100
        );
    }
    println!("</table></body></html>");
}

fn print_diff_extended(
    result: &diff::DiffResult,
    json: bool,
    format: &str,
    only_changes: bool,
    verbose: bool,
) -> Result<(), AppError> {
    if json {
        print_diff(result, true)?;
        return Ok(());
    }

    if format == "md" {
        print_diff_markdown(result, only_changes)?;
        return Ok(());
    }

    // Default text format with verdict header
    let status = if result.has_changes() {
        "DIFFERENT"
    } else {
        "IDENTICAL"
    };
    println!("{}", status);
    println!("{}", "-".repeat(20));

    if result.has_changes() {
        println!("Added tensors:    {}", result.tensors_added.len());
        println!("Removed tensors: {}", result.tensors_removed.len());
        println!("Modified tensors: {}", result.tensor_changes.len());
    } else {
        println!("No structural differences found.");
    }

    println!(
        "Structural ID:    {}",
        if result.hash_equal {
            "unchanged"
        } else {
            "changed"
        }
    );

    if !only_changes && !result.has_changes() {
        return Ok(());
    }

    if !result.tensor_changes.is_empty() && verbose {
        println!("\nTensor changes");
        println!("─────────────");
        for change in &result.tensor_changes {
            println!("  {}", change.name);
            if let (Some(old), Some(new)) = (&change.dtype_old, &change.dtype_new) {
                println!("    dtype: {} -> {}", old, new);
            }
            if let (Some(old), Some(new)) = (&change.shape_old, &change.shape_new) {
                println!("    shape: {:?} -> {:?}", old, new);
            }
        }
    }

    Ok(())
}

fn print_diff_markdown(result: &diff::DiffResult, only_changes: bool) -> Result<(), AppError> {
    let status = if result.has_changes() {
        "❌ DIFFERENT"
    } else {
        "✅ IDENTICAL"
    };
    println!("## Structural Diff");
    println!();
    println!("**Status:** {}", status);
    println!();
    println!("| Change Type | Count |");
    println!("|-------------|-------|");
    println!("| Added tensors | {} |", result.tensors_added.len());
    println!("| Removed tensors | {} |", result.tensors_removed.len());
    println!("| Modified tensors | {} |", result.tensor_changes.len());
    println!();

    if !only_changes
        && result.tensors_added.is_empty()
        && result.tensors_removed.is_empty()
        && result.tensor_changes.is_empty()
    {
        return Ok(());
    }

    if !result.tensors_added.is_empty() {
        println!("### Added tensors");
        println!("```");
        for name in &result.tensors_added {
            println!("+ {}", name);
        }
        println!("```");
    }

    if !result.tensors_removed.is_empty() {
        println!("### Removed tensors");
        println!("```");
        for name in &result.tensors_removed {
            println!("- {}", name);
        }
        println!("```");
    }

    if !result.tensor_changes.is_empty() {
        println!("### Modified tensors");
        println!("```");
        for change in &result.tensor_changes {
            println!("~ {}", change.name);
            if let (Some(old), Some(new)) = (&change.dtype_old, &change.dtype_new) {
                println!("  dtype: {} → {}", old, new);
            }
            if let (Some(old), Some(new)) = (&change.shape_old, &change.shape_new) {
                println!("  shape: {:?} → {:?}", old, new);
            }
        }
        println!("```");
    }

    Ok(())
}
