//! Integration tests for CLI commands
//! Run with: cargo test --test cli

use std::process::Command;
use tempfile::NamedTempFile;

/// Run the CLI with given arguments
fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new("cargo")
        .args(["run", "--"])
        .args(args)
        .current_dir("..")
        .output()
        .expect("Failed to run cargo")
}

#[test]
fn test_cli_help() {
    let output = run_cli(&["--help"]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("weight-inspect"));
    assert!(stdout.contains("inspect"));
    assert!(stdout.contains("id"));
    assert!(stdout.contains("diff"));
    assert!(stdout.contains("summary"));
}

#[test]
fn test_id_help() {
    let output = run_cli(&["id", "--help"]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("fingerprint"));
}

#[test]
fn test_inspect_help() {
    let output = run_cli(&["inspect", "--help"]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("structure"));
}

#[test]
fn test_diff_help() {
    let output = run_cli(&["diff", "--help"]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("compare"));
}

#[test]
fn test_summary_help() {
    let output = run_cli(&["summary", "--help"]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("scripts") || stdout.contains("One-line"));
}

#[test]
fn test_nonexistent_file() {
    let output = run_cli(&["inspect", "/nonexistent/file.gguf"]);

    // Should fail with non-zero exit code
    assert!(!output.status.success());
}

#[test]
fn test_id_json_flag() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let output = run_cli(&["id", &path.to_string_lossy(), "--json"]);

    // Just check the flag is accepted (should parse or fail gracefully)
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("stderr: {}", stderr);
}

#[test]
fn test_inspect_json_flag() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let output = run_cli(&["inspect", &path.to_string_lossy(), "--json"]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("stderr: {}", stderr);
}
