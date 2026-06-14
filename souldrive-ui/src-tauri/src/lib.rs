use std::{
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
};
use tauri::{Manager, Runtime};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

struct BackendProcess {
    child: Mutex<Option<Child>>,
    api_token: String,
    api_port: u16,
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                graceful_shutdown(&self.api_token, self.api_port);
                terminate_process_tree(&mut child);
            }
        }
    }
}

fn graceful_shutdown(api_token: &str, api_port: u16) {
    let url = format!("http://127.0.0.1:{}/shutdown", api_port);
    let client = reqwest::blocking::Client::new();
    let _ = client
        .post(&url)
        .header("X-SoulDrive-Token", api_token)
        .timeout(std::time::Duration::from_secs(5))
        .send();
    std::thread::sleep(std::time::Duration::from_millis(500));
}

#[cfg(windows)]
fn terminate_process_tree(child: &mut Child) {
    let _ = Command::new("taskkill")
        .args(["/PID", &child.id().to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .status();
    let _ = child.wait();
}

#[cfg(not(windows))]
fn terminate_process_tree(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[tauri::command]
fn backend_status(state: tauri::State<'_, BackendProcess>) -> bool {
    let Ok(mut guard) = state.child.lock() else {
        return false;
    };

    let Some(child) = guard.as_mut() else {
        return false;
    };

    match child.try_wait() {
        Ok(Some(_)) => {
            *guard = None;
            false
        }
        Ok(None) => true,
        Err(_) => false,
    }
}

#[derive(serde::Serialize)]
struct RuntimeConfig {
    base_url: String,
    api_token: String,
}

#[tauri::command]
fn runtime_config(state: tauri::State<'_, BackendProcess>) -> RuntimeConfig {
    RuntimeConfig {
        base_url: format!("http://127.0.0.1:{}", state.api_port),
        api_token: state.api_token.clone(),
    }
}

#[tauri::command]
fn select_pdf_files() -> Vec<String> {
    rfd::FileDialog::new()
        .add_filter("PDF", &["pdf"])
        .pick_files()
        .unwrap_or_default()
        .into_iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect()
}

fn spawn_backend<R: Runtime>(
    app: &impl Manager<R>,
    api_token: &str,
    api_port: u16,
) -> std::io::Result<Child> {
    let runtime_root = runtime_root(app);
    let configured_sidecar = sidecar_executable(app, &runtime_root);

    if let Some(sidecar_path) = configured_sidecar {
        let mut command = Command::new(sidecar_path);
        command
            .env("SOULDRIVE_API_TOKEN", api_token)
            .env("SOULDRIVE_API_PORT", api_port.to_string())
            .env("SOULDRIVE_APP_ROOT", &runtime_root)
            .env("SOULDRIVE_MODEL_DIR", runtime_root.join("models"))
            .env("SOULDRIVE_WATCH_REMOVABLE", "1")
            .env("SOULDRIVE_PARENT_PID", std::process::id().to_string())
            .current_dir(&runtime_root)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);

        return command.spawn();
    }

    let python = std::env::var("SOULDRIVE_PYTHON")
        .unwrap_or_else(|_| String::from("python"));

    let mut command = Command::new(python);
    command
        .arg("-m")
        .arg("core.sidecar_runtime")
        .env("SOULDRIVE_API_TOKEN", api_token)
        .env("SOULDRIVE_API_PORT", api_port.to_string())
        .env("SOULDRIVE_APP_ROOT", &runtime_root)
        .env("SOULDRIVE_MODEL_DIR", runtime_root.join("models"))
        .env("SOULDRIVE_WATCH_REMOVABLE", "1")
        .env("SOULDRIVE_PARENT_PID", std::process::id().to_string())
        .current_dir(&runtime_root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    #[cfg(windows)]
    command.creation_flags(CREATE_NO_WINDOW);

    command.spawn()
}

fn runtime_root<R: Runtime>(app: &impl Manager<R>) -> PathBuf {
    if let Ok(configured) = std::env::var("SOULDRIVE_APP_ROOT") {
        return PathBuf::from(configured);
    }

    for candidate in runtime_root_candidates(app) {
        if candidate.join("models").exists() || candidate.join("SoulDrive").exists() {
            return candidate;
        }
    }

    development_project_root()
}

fn sidecar_executable<R: Runtime>(app: &impl Manager<R>, runtime_root: &PathBuf) -> Option<PathBuf> {
    if let Ok(configured) = std::env::var("SOULDRIVE_SIDECAR_EXE") {
        let path = PathBuf::from(configured);
        if path.exists() {
            return Some(path);
        }
    }

    for candidate in sidecar_candidates(app, runtime_root) {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn sidecar_candidates<R: Runtime>(app: &impl Manager<R>, runtime_root: &PathBuf) -> Vec<PathBuf> {
    let executable_name = sidecar_executable_name();
    let mut roots = Vec::new();
    roots.push(runtime_root.clone());
    if let Ok(resource_dir) = app.path().resource_dir() {
        roots.push(resource_dir);
    }
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            roots.push(exe_dir.to_path_buf());
        }
    }

    let mut candidates = Vec::new();
    for root in roots {
        candidates.push(root.join("sidecars").join(executable_name));
        candidates.push(
            root.join("sidecars")
                .join("souldrive-sidecar")
                .join(executable_name),
        );
    }
    candidates
}

#[cfg(windows)]
fn sidecar_executable_name() -> &'static str {
    "souldrive-sidecar.exe"
}

#[cfg(not(windows))]
fn sidecar_executable_name() -> &'static str {
    "souldrive-sidecar"
}

fn runtime_root_candidates<R: Runtime>(app: &impl Manager<R>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir);
    }
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.to_path_buf());
        }
    }
    candidates.push(development_project_root());
    candidates
}

fn development_project_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .map(PathBuf::from)
        .unwrap_or(manifest_dir)
}

fn reserve_api_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let token = generate_api_token();
            let api_port = reserve_api_port().unwrap_or(8000);
            let child = match spawn_backend(app, &token, api_port) {
                Ok(child) => {
                    wait_for_sidecar_ready(&token, api_port);
                    Some(child)
                }
                Err(error) => {
                    eprintln!("failed to start SoulDrive sidecar runtime: {error}");
                    None
                }
            };
            app.manage(BackendProcess {
                child: Mutex::new(child),
                api_token: token.clone(),
                api_port,
            });

            // Start sidecar health monitor in background
            let token_clone = token.clone();
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                sidecar_health_monitor(handle, token_clone, api_port);
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![backend_status, runtime_config, select_pdf_files])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn wait_for_sidecar_ready(api_token: &str, api_port: u16) {
    let url = format!("http://127.0.0.1:{}/health", api_port);
    let client = reqwest::blocking::Client::new();
    for attempt in 0..30 {
        match client
            .get(&url)
            .header("X-SoulDrive-Token", api_token)
            .timeout(std::time::Duration::from_secs(2))
            .send()
        {
            Ok(response) if response.status().is_success() => {
                eprintln!("sidecar ready after {} attempts", attempt + 1);
                return;
            }
            _ => {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
    eprintln!("sidecar did not become ready within timeout");
}

fn sidecar_health_monitor(handle: tauri::AppHandle, api_token: String, api_port: u16) {
    let url = format!("http://127.0.0.1:{}/health", api_port);
    let client = reqwest::blocking::Client::new();
    let mut consecutive_failures = 0;
    loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        if restart_if_child_exited(&handle, &api_token, api_port) {
            consecutive_failures = 0;
            continue;
        }

        let healthy = client
            .get(&url)
            .header("X-SoulDrive-Token", &api_token)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        if healthy {
            consecutive_failures = 0;
            continue;
        }

        consecutive_failures += 1;
        eprintln!("sidecar health check failed ({consecutive_failures}/6)");
        if consecutive_failures >= 6 {
            restart_backend(&handle, &api_token, api_port, "health check failed repeatedly");
            consecutive_failures = 0;
        }
    }
}

fn restart_if_child_exited(handle: &tauri::AppHandle, api_token: &str, api_port: u16) -> bool {
    let state = handle.state::<BackendProcess>();
    let Ok(mut guard) = state.child.lock() else {
        return false;
    };

    let Some(child) = guard.as_mut() else {
        drop(guard);
        restart_backend(handle, api_token, api_port, "sidecar process missing");
        return true;
    };

    match child.try_wait() {
        Ok(Some(_)) => {
            *guard = None;
            drop(guard);
            restart_backend(handle, api_token, api_port, "sidecar process exited");
            true
        }
        Ok(None) => false,
        Err(error) => {
            eprintln!("sidecar process probe failed: {error}");
            false
        }
    }
}

fn restart_backend(handle: &tauri::AppHandle, api_token: &str, api_port: u16, reason: &str) {
    eprintln!("sidecar restart requested: {reason}");
    let state = handle.state::<BackendProcess>();
    let lock_result = state.child.lock();
    if let Ok(mut guard) = lock_result {
        if let Some(mut child) = guard.take() {
            terminate_process_tree(&mut child);
        }
        match spawn_backend(handle, api_token, api_port) {
            Ok(new_child) => {
                wait_for_sidecar_ready(api_token, api_port);
                *guard = Some(new_child);
                eprintln!("sidecar restarted successfully");
            }
            Err(error) => {
                eprintln!("sidecar restart failed: {error}");
            }
        }
    }
}

fn generate_api_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).expect("secure random API unavailable");
    let token = bytes.iter().map(|byte| format!("{:02x}", byte)).collect::<String>();
    format!("souldrive-{token}")
}
