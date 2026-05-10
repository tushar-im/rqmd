use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral")
        .local_addr()
        .expect("local addr")
        .port()
}

fn spawn_server(port: u16) -> Child {
    Command::new(env!("CARGO_BIN_EXE_rqmd"))
        .args(["serve", &format!("127.0.0.1:{port}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rqmd serve")
}

fn wait_for_server(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(25));
    }
    panic!("server did not start on port {port}");
}

fn http_request(port: u16, raw: &str) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream.write_all(raw.as_bytes()).expect("write request");
    stream.shutdown(std::net::Shutdown::Write).expect("shutdown write");

    let mut out = Vec::new();
    stream.read_to_end(&mut out).expect("read response");
    String::from_utf8(out).expect("utf8 response")
}

#[test]
fn rpc_health_and_query_contract() {
    let port = free_port();
    let mut child = spawn_server(port);
    wait_for_server(port);

    let health = http_request(
        port,
        "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert!(health.starts_with("HTTP/1.1 200 OK"), "{health}");
    assert!(health.contains("{\"status\":\"ok\"}"), "{health}");

    let query_body = "{\"query\":\"rust\",\"top_k\":2,\"corpus_dir\":\"eval/corpus\"}";
    let query = http_request(
        port,
        &format!(
            "POST /query HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            query_body.len(), query_body
        ),
    );
    assert!(query.starts_with("HTTP/1.1 200 OK"), "{query}");
    assert!(query.contains("\"results\""), "{query}");

    let missing_query_body = "{\"top_k\":2}";
    let missing_query_rsp = http_request(
        port,
        &format!(
            "POST /query HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            missing_query_body.len(), missing_query_body
        ),
    );
    assert!(missing_query_rsp.starts_with("HTTP/1.1 400 Bad Request"));

    child.kill().expect("kill server");
    let _ = child.wait();
}

#[test]
fn plugin_fixture_mentions_rpc_surface() {
    let status = fs::read_to_string("PORT_STATUS.md").expect("PORT_STATUS present");
    assert!(status.contains("GET /health"));
    assert!(status.contains("POST /query"));
}
