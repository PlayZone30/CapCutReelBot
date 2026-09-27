# CapCut Reel Automation — Final Plan

Confirmed against a real draft captured on macOS (CapCut 9.5.0). This is not a hypothetical schema — every field below was read out of an actual `draft_info.json` and `root_meta_info.json`.

---

## 1. What's confirmed

**File naming:** macOS uses `draft_info.json`. Windows likely uses `draft_content.json` (per community tooling) — unverified until we're on a Windows box; `config.py` will branch on this rather than assume.

**Segment schema (the timeline):**

- Each clip on the timeline is a `segment` inside `tracks[0]["segments"]`.
- `material_id` → points into `materials["videos"]` (path, duration, width, height).
- `source_timerange` = the trim *within* the source file (µs).
- `target_timerange` = position *on the timeline* (µs). Clips are placed back-to-back: clip 2's `start` equals clip 1's `duration`, exactly.
- `speed` on the segment is a float (1.0 = untouched). A matching entry also lives in `materials["speeds"]`, referenced via `extra_material_refs`.
- Every segment also drags along 5 more "empty" materials via `extra_material_refs`: one each from `placeholder_infos`, `canvases`, `material_colors`, `sound_channel_mappings`, `vocal_separations` — inert boilerplate present even when unused.
- Top-level `duration` = sum of every segment's `target_timerange.duration`. Verified exactly (589,066,666 = 320,300,000 + 268,766,666).
- Media is **not** copied into the draft folder — `materials.videos[].path` points straight at wherever the file already lives (your test draft pointed at `/Users/pavanreddy/Downloads/...`). Our rendered clips can stay in our own working folder.

**Registration schema (`root_meta_info.json`):**

- `all_draft_store` is a flat array; adding a draft = appending one entry.
- Its `draft_id` is independent from `draft_info.json`'s internal `id` — confirmed mismatched in your own file, so we don't need to keep them in sync.
- Key fields per entry: `draft_cover` (thumbnail jpg path), `draft_fold_path`, `draft_json_file`, `draft_name` (display name, can differ from folder name), `tm_draft_create` / `tm_draft_modified` (Unix microseconds), `tm_duration` (mirrors the draft's own `duration`).
- `draft_ids` at the top level tracks the count of entries.
- Skipping this step means the draft folder is technically valid but likely won't show up in CapCut's project list.

**Footage orientation:** your sample frame is genuinely portrait-composed (\~9:16 content, black pillarbox bars are just the screenshot framing, not baked into the video). So the default assumption is: no reframe/crop needed, canvas is native vertical. But this gets verified per-clip automatically rather than assumed globally (see §3) — phone footage occasionally carries a landscape pixel grid with a rotation flag instead of true vertical pixels, and that flag needs to survive re-encoding or the output comes out sideways.

---

## 2. Two decisions that shrink the risk

1. **Slow-motion is baked into the rendered file before it ever touches the JSON.** Every segment we write keeps `speed: 1.0` — we never generate a real speed-curve. The only reverse-engineered surface we depend on is "place a finished clip on the timeline," which is the most stable, well-trodden part of this schema.
2. **We clone your captured draft rather than generating one from scratch.** The boilerplate (canvas config, HDR settings, enable-flags, the 5 empty materials per segment) gets copied from your real template and only the parts we care about — the clip list — get replaced.

---

## 3. Pipeline

```
Drive folder
  → drive_ingest.py    pulls raw clips to /raw
  → probe.py           ffprobe: fps, duration, resolution, rotation tag per clip
                        → if rotation metadata present, normalize orientation now
                          so every downstream step works on correctly-oriented pixels
  → gemini_score.py    Gemini (Flash, native video input) scores each clip against
                        your rubric, returns best in/out ranges + reasoning
  → speed_process.py   effective_fps = source_fps × speed_factor
                          < ~24  → RIFE frame interpolation
                          else   → plain `setpts` conform
  → reframe.py         only runs if a clip's *actual* pixel grid (post-rotation-fix)
                        is landscape; center-crop to 1080x1920. Skipped entirely
                        for clips that are already vertical (expected default,
                        per your sample frame)
  → draft_writer.py    clone_draft() → add_clip_to_draft() per surviving clip
                        → register_draft() appends to root_meta_info.json
  → you open the draft in CapCut for your human pass
```

---

## 4. `draft_writer.py` — the two core functions

```python
def add_clip_to_draft(draft, clip_path, duration_us, width, height):
    video_id = new_uuid()
    draft["materials"]["videos"].append({
        "id": video_id, "material_name": os.path.basename(clip_path),
        "path": clip_path, "duration": duration_us,
        "width": width, "height": height, "type": "video"
    })

    speed_id  = add_empty(draft, "speeds", speed=1.0, mode=0, curve_speed=None, type="speed")
    ph_id     = add_empty(draft, "placeholder_infos", meta_type="none", type="placeholder_info")
    canvas_id = add_empty(draft, "canvases", type="canvas_color")
    color_id  = add_empty(draft, "material_colors")
    sound_id  = add_empty(draft, "sound_channel_mappings", audio_channel_mapping=0)
    vocal_id  = add_empty(draft, "vocal_separations", choice=0, type="vocal_separation")

    start = draft["duration"]
    segment = clone_template_segment()   # copies clip/enable_*/hdr_settings verbatim
    segment.update({
        "id": new_uuid(),
        "material_id": video_id,
        "extra_material_refs": [speed_id, ph_id, canvas_id, color_id, sound_id, vocal_id],
        "source_timerange": {"start": 0, "duration": duration_us},
        "target_timerange": {"start": start, "duration": duration_us},
        "speed": 1.0,
    })
    draft["tracks"][0]["segments"].append(segment)
    draft["duration"] += duration_us


def register_draft(root_meta_path, folder_path, draft_json_path, draft_name, duration_us):
    with open(root_meta_path) as f:
        root = json.load(f)
    now_us = int(time.time() * 1_000_000)
    root["all_draft_store"].append({
        "cloud_draft_cover": False, "cloud_draft_sync": False,
        "draft_cloud_last_action_download": False, "draft_cloud_purchase_info": "",
        "draft_cloud_template_id": "", "draft_cloud_tutorial_info": "",
        "draft_cloud_videocut_purchase_info": "",
        "draft_cover": os.path.join(folder_path, "draft_cover.jpg"),
        "draft_fold_path": folder_path,
        "draft_id": new_uuid(),
        "draft_is_ai_shorts": False, "draft_is_cloud_temp_draft": False,
        "draft_is_infinite_canvas_draft": False, "draft_is_invisible": False,
        "draft_is_pippit_draft": False, "draft_is_web_article_video": False,
        "draft_json_file": draft_json_path, "draft_name": draft_name,
        "draft_new_version": "", "draft_root_path": os.path.dirname(folder_path),
        "draft_timeline_materials_size": 0, "draft_type": "",
        "draft_web_article_video_enter_from": "",
        "pippit_avatar_url": "", "pippit_extra_info": "", "pippit_id": "", "pippit_user_name": "",
        "streaming_edit_draft_ready": True,
        "tm_draft_cloud_completed": "", "tm_draft_cloud_entry_id": -1,
        "tm_draft_cloud_modified": 0, "tm_draft_cloud_parent_entry_id": -1,
        "tm_draft_cloud_space_id": -1, "tm_draft_cloud_user_id": -1,
        "tm_draft_create": now_us, "tm_draft_modified": now_us,
        "tm_draft_removed": 0, "tm_duration": duration_us,
    })
    root["draft_ids"] = len(root["all_draft_store"])
    with open(root_meta_path, "w") as f:
        json.dump(root, f)
```

A thumbnail (`draft_cover.jpg`) is cosmetic only — worth generating from the first clip's first frame via ffmpeg eventually, not a blocker.

---

## 5. Repo layout

```
capcut_reel_bot/
  config.py            # OS detection, CAPCUT_DRAFT_ROOT, DRAFT_FILENAME
  drive_ingest.py       # Google Drive → /raw
  probe.py              # ffprobe: fps/duration/res/rotation per clip
  gemini_score.py       # Gemini Flash → best in/out ranges per clip
  speed_process.py      # setpts vs RIFE branch → /processed
  reframe.py            # only touches clips whose real pixel grid is landscape
  draft_writer.py        # clone_draft(), add_clip_to_draft(), register_draft()
  templates/
    reference_draft/     # your captured "0926" folder, sanitized, as the clone source
  main.py                # orchestrates the batch end to end
```

## 6. What's needed to start building

- `GEMINI_API_KEY`
- Google Drive folder ID + credentials (service account or OAuth)
- Confirmation once you're on the Windows machine that `draft_content.json` there matches this same field shape under a different filename

## 7. Still open, not blocking

- Windows schema parity (verify when available)
- Whether any of your real clips carry rotation metadata rather than true vertical pixels — `probe.py`'s check handles either case automatically, so this doesn't change the plan, just which branch fires per clip