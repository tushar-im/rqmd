use std::fs;
use std::process::{Command, Stdio};

#[test]
fn plugin_fixture_schema_is_replayable() {
    let body = fs::read_to_string("fixtures/plugin_calls.jsonl").expect("fixture exists");
    let first = body.lines().next().expect("line exists");
    assert!(first.contains("request"));
    assert!(first.contains("response"));
}

#[test]
fn plugin_fixture_replay_through_cli_contract() {
    let line = fs::read_to_string("fixtures/plugin_calls.jsonl").expect("fixture exists");
    let first = line.lines().next().expect("line exists");
    let request = first
        .split("\"request\":")
        .nth(1)
        .and_then(|s| s.split(",\"response\":").next())
        .expect("request object")
        .to_string();

    let mut child = Command::new(env!("CARGO_BIN_EXE_rqmd"))
        .arg("plugin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn rqmd");

    use std::io::Write;
    child.stdin.as_mut().expect("stdin").write_all(request.as_bytes()).expect("write stdin");

    let out = child.wait_with_output().expect("wait output");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"results\""));
}
