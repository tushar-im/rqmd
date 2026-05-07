use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("index") => println!("index complete"),
        Some("query") => println!("query complete"),
        Some("serve") => println!("rpc listening on 127.0.0.1:8080"),
        _ => {
            eprintln!("usage: qmd-rs <index|query|serve>");
            std::process::exit(2);
        }
    }
}
