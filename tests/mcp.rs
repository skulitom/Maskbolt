//! MCP round trip in process (duplex pipe) with the fake backend: list tools, open, click, find, view, gaps,
//! fill holes, remove, export, and the root guard.
mod common;
use common::*;
use maskbolt::mcp::Maskbolt;
use maskbolt::FakeBackend;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::ServiceExt;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

fn args(v: Value) -> serde_json::Map<String, Value> {
    v.as_object().unwrap().clone()
}

fn reply(r: &CallToolResult) -> Value {
    let t = r.content.iter().find_map(|c| c.as_text()).expect("a text block");
    serde_json::from_str(&t.text).unwrap()
}

fn images(r: &CallToolResult) -> usize {
    r.content.iter().filter(|c| c.as_image().is_some()).count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_round_trip_with_fake_backend() {
    let root = temp_dir("mcp");
    let mut img = ring_scene();
    disc(&mut img, 30, 140, 12, BLUE);
    img.save(root.join("scene.png")).unwrap();
    let outside = root.parent().unwrap().join(format!("{}-outside.png", root.file_name().unwrap().to_string_lossy()));
    img.save(&outside).unwrap();

    let server = Maskbolt::with_backend(&root, Arc::new(FakeBackend::default())).unwrap();
    let (client_io, server_io) = tokio::io::duplex(1 << 22);
    let server_task = tokio::spawn(async move {
        let svc = server.serve(server_io).await.unwrap();
        let _ = svc.waiting().await;
    });
    tokio::time::timeout(Duration::from_secs(50), async {
        let client = ().serve(client_io).await.unwrap();
        let info = client.peer_info().expect("server info");
        assert_eq!(info.server_info.as_ref().map(|s| s.name.as_str()), Some("maskbolt"));
        assert!(info.instructions.as_deref().unwrap_or("").contains("maskbolt_open"));

        let mut names: Vec<String> = client.list_all_tools().await.unwrap().into_iter().map(|t| t.name.to_string()).collect();
        names.sort();
        assert_eq!(
            names,
            ["maskbolt_click", "maskbolt_export", "maskbolt_fill_holes", "maskbolt_find", "maskbolt_gaps", "maskbolt_open", "maskbolt_remove", "maskbolt_view"]
        );
        let call = |name: &'static str, v: Value| {
            let client = &client;
            async move { client.call_tool(CallToolRequestParams::new(name).with_arguments(args(v))).await.unwrap() }
        };

        // Before open: a clear error.
        let r = call("maskbolt_view", json!({})).await;
        assert_eq!(r.is_error, Some(true));
        assert!(reply(&r)["error"].as_str().unwrap().contains("maskbolt_open"));

        let r = call("maskbolt_open", json!({"path": "scene.png"})).await;
        assert_ne!(r.is_error, Some(true), "{r:?}");
        let v = reply(&r);
        assert_eq!(v["opened"]["width"], 200);
        assert_eq!(v["opened"]["model"], "fake");

        let r = call("maskbolt_click", json!({"points": [[50, 80]]})).await;
        let v = reply(&r);
        assert_eq!(v["click"]["id"], 1, "{v}");
        assert_eq!(v["click"]["area"], 5500);
        assert_eq!(v["status"]["masks"], 1);
        assert!(v["status"]["still_uncovered"]["hint"].is_string());
        assert_eq!(images(&r), 1, "click replies carry a zoomed preview");

        // An exclude point and a level.
        let r = call("maskbolt_click", json!({"points": [[160, 50]], "exclude": [[5, 5]], "level": "largest"})).await;
        assert_eq!(reply(&r)["click"]["id"], 2);

        let r = call("maskbolt_view", json!({"region": {"x": 0, "y": 0, "w": 100, "h": 100}})).await;
        let v = reply(&r);
        assert_eq!(v["total"], 2);
        assert_eq!(v["listed"], 1, "only the ring overlaps the region: {v}");
        assert_eq!(images(&r), 1);

        let r = call("maskbolt_gaps", json!({})).await;
        let v = reply(&r);
        assert!(v["candidates"].as_u64().is_some(), "{v}");

        let r = call("maskbolt_find", json!({"grid": 8})).await;
        let v = reply(&r);
        assert!(v["find"]["prompts"] == 64 && v["find"]["dropped"].is_object(), "{v}");
        assert_eq!(v["status"]["masks"], 3, "find adds the small blue disc: {v}");
        assert_eq!(images(&r), 1);

        let r = call("maskbolt_fill_holes", json!({"ids": [1]})).await;
        assert_eq!(reply(&r)["filled"][0]["filled"], 900);

        let r = call("maskbolt_remove", json!({"ids": [3, 99]})).await;
        let v = reply(&r);
        assert_eq!(v["removed"], json!([3]));
        assert_eq!(v["not_found"], json!([99]));

        for (format, path) in [("cutout", "out/ring.png"), ("json", "out/masks.json"), ("svg", "out/masks.svg"), ("overlay", "out/overlay.png"), ("labels", "out/labels.png")] {
            let r = call("maskbolt_export", json!({"format": format, "path": path, "mm_per_px": if format == "svg" { json!(0.5) } else { Value::Null }})).await;
            assert_ne!(r.is_error, Some(true), "{format}: {r:?}");
            assert!(root.join(path).is_file(), "{path}");
        }
        let cut = image::open(root.join("out/ring.png")).unwrap();
        assert_eq!((cut.width(), cut.height()), (136, 85), "union of the ring (40..119) and the disc (145..175 x 35..65)");
        let j: Value = serde_json::from_str(&std::fs::read_to_string(root.join("out/masks.json")).unwrap()).unwrap();
        assert_eq!(j["count"], 2);

        // The root guard.
        let r = call("maskbolt_export", json!({"format": "json", "path": "../escape.json"})).await;
        assert_eq!(r.is_error, Some(true));
        assert!(reply(&r)["error"].as_str().unwrap().contains("outside the allowed root"));
        let r = call("maskbolt_open", json!({"path": outside.to_str().unwrap()})).await;
        assert_eq!(r.is_error, Some(true));
        assert!(reply(&r)["error"].as_str().unwrap().contains("outside the allowed root"));
        // Bad arguments are reported, not crashes.
        let r = call("maskbolt_click", json!({"points": [[5000, 5]]})).await;
        assert_eq!(r.is_error, Some(true));

        client.cancel().await.unwrap();
    })
    .await
    .expect("MCP round trip timed out");
    server_task.abort();
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_file(&outside).ok();
}

// Review P2-3 over MCP: export never replaces the open image, even with overwrite: true, and refuses other
// existing files unless overwrite is true.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn p2_3_mcp_export_protects_existing_files() {
    let root = temp_dir("mcp-p23");
    ring_scene().save(root.join("scene.png")).unwrap();
    let original = std::fs::read(root.join("scene.png")).unwrap();
    std::fs::write(root.join("keep.json"), b"mine").unwrap();
    let server = Maskbolt::with_backend(&root, Arc::new(FakeBackend::default())).unwrap();
    let (client_io, server_io) = tokio::io::duplex(1 << 22);
    let server_task = tokio::spawn(async move {
        let svc = server.serve(server_io).await.unwrap();
        let _ = svc.waiting().await;
    });
    tokio::time::timeout(Duration::from_secs(50), async {
        let client = ().serve(client_io).await.unwrap();
        let call = |name: &'static str, v: Value| {
            let client = &client;
            async move { client.call_tool(CallToolRequestParams::new(name).with_arguments(args(v))).await.unwrap() }
        };
        call("maskbolt_open", json!({"path": "scene.png"})).await;
        call("maskbolt_click", json!({"points": [[50, 80]]})).await;
        let r = call("maskbolt_export", json!({"format": "overlay", "path": "scene.png", "overwrite": true})).await;
        assert_eq!(r.is_error, Some(true), "{r:?}");
        assert_eq!(std::fs::read(root.join("scene.png")).unwrap(), original, "the open image was replaced");
        let r = call("maskbolt_export", json!({"format": "json", "path": "keep.json"})).await;
        assert_eq!(r.is_error, Some(true));
        assert!(reply(&r)["error"].as_str().unwrap().contains("overwrite"));
        assert_eq!(std::fs::read(root.join("keep.json")).unwrap(), b"mine");
        let r = call("maskbolt_export", json!({"format": "json", "path": "keep.json", "overwrite": true})).await;
        assert_ne!(r.is_error, Some(true), "{r:?}");
        client.cancel().await.unwrap();
    })
    .await
    .expect("timed out");
    server_task.abort();
    std::fs::remove_dir_all(&root).ok();
}
