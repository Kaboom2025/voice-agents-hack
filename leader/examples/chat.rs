//! Interactive CLI chat with Gemma 4 via cactus-sys.
//! Usage: cargo run --release --package leader --example chat

use std::ffi::{CStr, CString};
use std::io::{self, BufRead, Write};
use std::ptr;

fn main() {
    let model_path =
        std::env::var("CACTUS_MODEL_PATH").unwrap_or_else(|_| "weights/gemma-4-e2b-it".to_string());

    eprintln!("loading model from {model_path} …");
    let path_c = CString::new(model_path.as_str()).unwrap();
    let model = unsafe { cactus_sys::cactus_init(path_c.as_ptr(), ptr::null(), false) };
    assert!(!model.is_null(), "cactus_init returned null");
    eprintln!("model loaded — type your questions (Ctrl-D to quit)\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("you> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }
        let question = line.trim();
        if question.is_empty() {
            continue;
        }

        let messages = format!(
            r#"[{{"role":"user","content":"{}"}}]"#,
            question.replace('\\', "\\\\").replace('"', "\\\"")
        );
        let options = r#"{"max_tokens":256}"#;

        let msgs_c = CString::new(messages).unwrap();
        let opts_c = CString::new(options).unwrap();
        let mut buf = vec![0i8; 32 * 1024];

        let rc = unsafe {
            cactus_sys::cactus_complete(
                model,
                msgs_c.as_ptr(),
                buf.as_mut_ptr(),
                buf.len(),
                opts_c.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                ptr::null(),
                0,
            )
        };

        if rc <= 0 {
            eprintln!("(cactus_complete returned {rc})");
            continue;
        }

        let raw = unsafe { CStr::from_ptr(buf.as_ptr()).to_string_lossy() };

        // Try to extract just the "response" field for clean output.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            let answer = v["response"].as_str().unwrap_or(&raw);
            let ttft = v["time_to_first_token_ms"].as_f64().unwrap_or(0.0);
            let total = v["total_time_ms"].as_f64().unwrap_or(0.0);
            let tps = v["decode_tps"].as_f64().unwrap_or(0.0);
            println!("\ngemma4> {answer}");
            eprintln!("  [{ttft:.0}ms ttft, {total:.0}ms total, {tps:.0} tok/s]\n");
        } else {
            println!("\ngemma4> {raw}\n");
        }
    }

    unsafe { cactus_sys::cactus_destroy(model) };
    eprintln!("bye");
}
