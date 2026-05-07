use std::fs;

#[test]
fn plugin_fixture_schema_is_replayable() {
    let body = fs::read_to_string("fixtures/plugin_calls.jsonl").expect("fixture exists");
    let first = body.lines().next().expect("line exists");
    assert!(first.contains("request"));
    assert!(first.contains("response"));
}
