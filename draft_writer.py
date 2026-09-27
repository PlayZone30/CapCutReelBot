"""
draft_writer.py — clone the reference CapCut draft and append clips to it.

Everything here is built directly against a real captured
`draft_info.json` / `root_meta_info.json` pair (see implementationplan.md
for the field-by-field notes). We do not generate a draft from scratch —
we clone `templates/reference_draft/`, strip its two sample segments, and
then append real clips one at a time with `add_clip_to_draft`.

Key confirmed facts this module relies on:
- durations/timeranges are in microseconds.
- clips are placed back-to-back on tracks[0]["segments"]: clip N's
  target_timerange.start == sum of all previous clips' durations.
- every segment drags along 5 "empty" materials via extra_material_refs:
  one each from speeds, placeholder_infos, canvases, material_colors,
  sound_channel_mappings, vocal_separations.
- top-level `duration` = sum of every segment's target_timerange.duration.

Media placement (revised from the original plan):
- CapCut on macOS is a sandboxed app (com.apple.security.app-sandbox). It
  can only read files that live somewhere it's been explicitly granted
  access to — its own draft folders under ~/Movies/CapCut/..., or files
  the user picked via a native file dialog. It CANNOT read arbitrary
  paths like ~/capcut_reel_bot/processed/, even with Full Disk Access
  granted in System Settings (confirmed by hands-on testing — clips
  showed "Unsupported Media" / "File not accessible" in the CapCut UI).
- To work around this, build_draft() now copies each clip into
  <draft_folder>/materials/ (inside CapCut's own sandbox container) and
  points materials.videos[].path at that copy, instead of referencing
  the file in place under this repo. This is why raw/processed clips
  living under the repo now show up correctly in CapCut.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import config

# The 6 "empty" material buckets every segment drags along via
# extra_material_refs, and the boilerplate fields each entry needs.
_EMPTY_MATERIAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "speeds": {"curve_speed": None, "mode": 0, "speed": 1.0, "type": "speed"},
    "placeholder_infos": {
        "error_path": "", "error_text": "", "meta_type": "none",
        "res_path": "", "res_text": "", "type": "placeholder_info",
    },
    "canvases": {
        "album_image": "", "blur": 0.0, "color": "", "image": "", "image_id": "",
        "image_name": "", "source_platform": 0, "team_id": "", "type": "canvas_color",
    },
    "material_colors": {
        "gradient_angle": 90.0, "gradient_colors": [], "gradient_percents": [],
        "height": 0.0, "is_color_clip": False, "is_gradient": False,
        "solid_color": "", "width": 0.0,
    },
    "sound_channel_mappings": {"audio_channel_mapping": 0, "is_config_open": False, "type": ""},
    "vocal_separations": {
        "choice": 0, "enter_from": "", "final_algorithm": "", "production_path": "",
        "removed_sounds": [], "time_range": None, "type": "vocal_separation",
    },
}

# extra_material_refs order, confirmed from the captured draft:
# [speed, placeholder_info, canvas, material_color, sound_channel_mapping, vocal_separation]
_REF_ORDER = ["speeds", "placeholder_infos", "canvases", "material_colors",
              "sound_channel_mappings", "vocal_separations"]

# Segment template fields copied verbatim from the captured draft (clip/
# enable_*/hdr_settings/etc). Only the fields under "overridden below" get
# replaced per-clip in add_clip_to_draft.
_SEGMENT_TEMPLATE: dict[str, Any] = {
    "caption_info": None,
    "cartoon": False,
    "clip": {
        "alpha": 1.0,
        "flip": {"horizontal": False, "vertical": False},
        "rotation": 0.0,
        "scale": {"x": 1.0, "y": 1.0},
        "transform": {"x": 0.0, "y": 0.0},
    },
    "color_correct_alg_result": "",
    "common_keyframes": [],
    "desc": "",
    "digital_human_template_group_id": "",
    "enable_adjust": True,
    "enable_adjust_mask": False,
    "enable_color_adjust_pro": False,
    "enable_color_correct_adjust": False,
    "enable_color_curves": True,
    "enable_color_match_adjust": False,
    "enable_color_wheels": True,
    "enable_hsl": False,
    "enable_hsl_curves": True,
    "enable_lut": True,
    "enable_mask_shadow": False,
    "enable_mask_stroke": False,
    "enable_smart_color_adjust": False,
    "enable_video_mask": True,
    "group_id": "",
    "hdr_settings": {"intensity": 1.0, "mode": 1, "nits": 1000},
    "hdr_vivid_settings": None,
    "intensifies_audio": False,
    "is_loop": False,
    "is_placeholder": False,
    "is_tone_modify": False,
    "keyframe_refs": [],
    "last_nonzero_volume": 1.0,
    "lyric_keyframes": None,
    "raw_segment_id": "",
    "render_index": 0,
    "render_timerange": {"duration": 0, "start": 0},
    "responsive_layout": {
        "enable": False, "horizontal_pos_layout": 0, "size_layout": 0,
        "target_follow": "", "vertical_pos_layout": 0,
    },
    "reverse": False,
    "segment_color_tag": "",
    "source": "segmentsourcenormal",
    "state": 0,
    "template_id": "",
    "template_scene": "default",
    "track_attribute": 0,
    "track_render_index": 0,
    "uniform_scale": {"on": True, "value": 1.0},
    "visible": True,
    "volume": 1.0,
}

# video material template fields copied verbatim, minus the per-clip
# overrides applied in add_clip_to_draft.
_VIDEO_MATERIAL_TEMPLATE: dict[str, Any] = {
    "aigc_history_id": "", "aigc_item_id": "", "aigc_type": "none", "audio_fade": None,
    "beauty_body_auto_preset": None, "beauty_body_preset_id": "",
    "beauty_face_auto_preset": {"name": "", "preset_id": "", "rate_map": "", "scene": ""},
    "beauty_face_auto_preset_infos": [], "beauty_face_preset_infos": [],
    "cartoon_path": "", "category_id": "", "category_name": "local",
    "check_flag": 62978047, "content_feature_info": None, "corner_pin": None,
    "crop": {
        "lower_left_x": 0.0, "lower_left_y": 1.0, "lower_right_x": 1.0, "lower_right_y": 1.0,
        "upper_left_x": 0.0, "upper_left_y": 0.0, "upper_right_x": 1.0, "upper_right_y": 0.0,
    },
    "crop_ratio": "free", "crop_scale": 1.0, "extra_type_option": 0, "formula_id": "",
    "freeze": None, "has_audio": True, "has_sound_separated": False,
    "intensifies_audio_path": "", "intensifies_path": "", "is_ai_generate_content": False,
    "is_copyright": False, "is_set_beauty_mode": False, "is_text_edit_overdub": False,
    "is_unified_beauty_mode": False, "is_video_copilot_aigc_content": False,
    "live_photo_cover_path": "", "live_photo_timestamp": -1, "local_id": "",
    "local_material_from": "", "material_id": "", "material_url": "",
    "matting": {
        "cloud_product_fps": 0.0, "custom_matting_id": "", "enable_matting_stroke": False,
        "expansion": 0, "feather": 0, "flag": 0, "has_use_quick_brush": False,
        "has_use_quick_eraser": False, "interactiveTime": [], "is_clould": False,
        "mask_video_path": "", "path": "", "reverse": False, "strokes": [],
    },
    "media_path": "", "multi_camera_info": None, "object_locked": None,
    "origin_material_id": "", "picture_from": "none", "picture_set_category_id": "",
    "picture_set_category_name": "", "pre_applied_vip_materials": [], "request_id": "",
    "reverse_intensifies_path": "", "reverse_path": "", "smart_match_info": None,
    "smart_motion": None, "source": 0, "source_platform": 0,
    "stable": {"matrix_path": "", "stable_level": 0, "time_range": {"duration": 0, "start": 0}},
    "surface_trackings": [], "team_id": "", "type": "video", "unique_id": "",
    "video_algorithm": {
        "ai_background_configs": [], "ai_expression_driven": None, "ai_in_painting_config": [],
        "ai_motion_driven": None, "aigc_generate": None, "aigc_generate_list": [],
        "algorithms": [], "complement_frame_config": None, "deflicker": None,
        "gameplay_configs": [], "image_interpretation": None, "motion_blur_config": None,
        "mouth_shape_driver": None, "noise_reduction": None, "path": "",
        "quality_enhance": None, "skip_algorithm_index": [], "smart_complement_frame": None,
        "story_video_modify_video_config": {
            "generate_card_id": "", "generate_id": "", "is_overwrite_last_video": False,
            "task_id": "", "tracker_task_id": "",
        },
        "super_resolution": None, "time_range": None,
    },
    "video_mask_shadow": {
        "alpha": 0.0, "angle": 0.0, "blur": 0.0, "color": "", "distance": 0.0,
        "path": "", "resource_id": "",
    },
    "video_mask_stroke": {
        "alpha": 0.0, "color": "", "distance": 0.0, "horizontal_shift": 0.0,
        "path": "", "resource_id": "", "size": 0.0, "texture": 0.0, "type": "",
        "vertical_shift": 0.0,
    },
    "workflow_node_id": "",
}


def new_uuid() -> str:
    """CapCut ids are uppercase hyphenated UUIDs, e.g. B57F1051-E672-482A-987C-E542D81CE4B6."""
    return str(uuid.uuid4()).upper()


def clone_draft(draft_name: str, dest_root: Path | str | None = None,
                 template_dir: Path | str | None = None) -> Path:
    """
    Clone templates/reference_draft/ into <dest_root>/<draft_name>, strip the
    two sample segments/materials from the cloned draft_info.json so it
    starts as an empty timeline, and return the new draft folder path.
    """
    template_dir = Path(template_dir or config.TEMPLATE_DRAFT_DIR)
    dest_root = Path(dest_root or config.CAPCUT_DRAFT_ROOT)
    dest_folder = dest_root / draft_name

    if dest_folder.exists():
        raise FileExistsError(f"Draft folder already exists: {dest_folder}")

    shutil.copytree(template_dir, dest_folder)

    draft_path = dest_folder / config.DRAFT_FILENAME
    with open(draft_path, encoding="utf-8") as f:
        draft = json.load(f)

    # Reset to an empty timeline: wipe segments + all per-clip materials,
    # keep the top-level boilerplate (canvas_config, config, platform, etc).
    draft["id"] = new_uuid()
    draft["duration"] = 0
    for track in draft.get("tracks", []):
        track["segments"] = []
    for bucket in _REF_ORDER + ["videos"]:
        draft["materials"][bucket] = []

    with open(draft_path, "w", encoding="utf-8") as f:
        json.dump(draft, f)

    return dest_folder


def _add_empty(draft: dict, bucket: str, **overrides: Any) -> str:
    """Append a new entry to materials[bucket] using the confirmed defaults, return its id."""
    entry = copy.deepcopy(_EMPTY_MATERIAL_DEFAULTS[bucket])
    entry.update(overrides)
    entry["id"] = new_uuid()
    draft["materials"][bucket].append(entry)
    return entry["id"]


def _clone_template_segment() -> dict:
    return copy.deepcopy(_SEGMENT_TEMPLATE)


def add_clip_to_draft(draft: dict, clip_path: str | Path, duration_us: int,
                       width: int, height: int, *, track_index: int = 0) -> dict:
    """
    Append one clip to the draft's timeline (in memory). Mutates `draft`
    and returns the segment dict that was added.

    speed is always 1.0 here — slow motion is baked into the rendered file
    by speed_process.py before it ever reaches this function.
    """
    video_id = new_uuid()
    video_material = copy.deepcopy(_VIDEO_MATERIAL_TEMPLATE)
    video_material.update({
        "id": video_id,
        "local_material_id": str(uuid.uuid4()),
        "material_name": os.path.basename(str(clip_path)),
        "path": str(clip_path),
        "duration": duration_us,
        "width": width,
        "height": height,
    })
    draft["materials"]["videos"].append(video_material)

    speed_id = _add_empty(draft, "speeds")
    ph_id = _add_empty(draft, "placeholder_infos")
    canvas_id = _add_empty(draft, "canvases")
    color_id = _add_empty(draft, "material_colors")
    sound_id = _add_empty(draft, "sound_channel_mappings")
    vocal_id = _add_empty(draft, "vocal_separations")

    start = draft["duration"]
    segment = _clone_template_segment()
    segment.update({
        "id": new_uuid(),
        "material_id": video_id,
        "extra_material_refs": [speed_id, ph_id, canvas_id, color_id, sound_id, vocal_id],
        "source_timerange": {"start": 0, "duration": duration_us},
        "target_timerange": {"start": start, "duration": duration_us},
        "speed": 1.0,
    })

    draft["tracks"][track_index]["segments"].append(segment)
    draft["duration"] += duration_us
    return segment


def save_draft(draft: dict, draft_folder: Path | str) -> Path:
    draft_path = Path(draft_folder) / config.DRAFT_FILENAME
    with open(draft_path, "w", encoding="utf-8") as f:
        json.dump(draft, f)
    return draft_path


def load_draft(draft_folder: Path | str) -> dict:
    draft_path = Path(draft_folder) / config.DRAFT_FILENAME
    with open(draft_path, encoding="utf-8") as f:
        return json.load(f)


def register_draft(root_meta_path: Path | str, folder_path: Path | str,
                    draft_json_path: Path | str, draft_name: str, duration_us: int) -> None:
    """
    Append one entry to root_meta_info.json's all_draft_store, and bump
    draft_ids. Without this the draft folder is valid but won't show up in
    CapCut's project list.
    """
    root_meta_path = Path(root_meta_path)
    with open(root_meta_path, encoding="utf-8") as f:
        root = json.load(f)

    now_us = int(time.time() * 1_000_000)
    root["all_draft_store"].append({
        "cloud_draft_cover": False, "cloud_draft_sync": False,
        "draft_cloud_last_action_download": False, "draft_cloud_purchase_info": "",
        "draft_cloud_template_id": "", "draft_cloud_tutorial_info": "",
        "draft_cloud_videocut_purchase_info": "",
        "draft_cover": str(Path(folder_path) / "draft_cover.jpg"),
        "draft_fold_path": str(folder_path),
        "draft_id": new_uuid(),
        "draft_is_ai_shorts": False, "draft_is_cloud_temp_draft": False,
        "draft_is_infinite_canvas_draft": False, "draft_is_invisible": False,
        "draft_is_pippit_draft": False, "draft_is_web_article_video": False,
        "draft_json_file": str(draft_json_path), "draft_name": draft_name,
        "draft_new_version": "", "draft_root_path": str(Path(folder_path).parent),
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

    with open(root_meta_path, "w", encoding="utf-8") as f:
        json.dump(root, f)


def _copy_into_draft_materials(clip_path: str | Path, dest_folder: Path | str) -> Path:
    """
    Copy a clip into <dest_folder>/materials/ and return the new path.

    Required because CapCut's macOS sandbox can only read files inside its
    own draft folder (or files the user picked via a native dialog) — see
    the module docstring. Referencing a clip in place under this repo
    results in "Unsupported Media" in the CapCut UI even with Full Disk
    Access granted, confirmed by hands-on testing.
    """
    clip_path = Path(clip_path)
    materials_dir = Path(dest_folder) / "materials"
    materials_dir.mkdir(parents=True, exist_ok=True)
    dest_path = materials_dir / clip_path.name
    shutil.copy2(clip_path, dest_path)
    return dest_path


def build_draft(draft_name: str, clips: list[dict[str, Any]],
                 root_meta_path: Path | str, dest_root: Path | str | None = None) -> Path:
    """
    End-to-end helper: clone the template, copy every clip into the draft's
    own materials/ folder (so CapCut's sandbox can read them), append them
    to the timeline, save, and register the draft so it shows up in CapCut.

    `clips` is a list of dicts with keys: path, duration_us, width, height.
    """
    dest_folder = clone_draft(draft_name, dest_root=dest_root)
    draft = load_draft(dest_folder)

    for clip in clips:
        in_draft_path = _copy_into_draft_materials(clip["path"], dest_folder)
        add_clip_to_draft(draft, in_draft_path, clip["duration_us"], clip["width"], clip["height"])

    save_draft(draft, dest_folder)
    register_draft(
        root_meta_path=root_meta_path,
        folder_path=dest_folder,
        draft_json_path=dest_folder / config.DRAFT_FILENAME,
        draft_name=draft_name,
        duration_us=draft["duration"],
    )
    return dest_folder


if __name__ == "__main__":
    # Smoke test using the two real sample clips referenced in the captured
    # draft. Writes into a throwaway draft name so it doesn't collide with
    # the real "0926" draft.
    from probe import probe_clip

    sample_clips = [
        Path.home() / "Downloads" / "3rules.mp4",
        Path.home() / "Downloads" / "example.mp4",
    ]
    existing = [p for p in sample_clips if p.exists()]
    if not existing:
        print("No sample clips found under ~/Downloads; skipping smoke test.")
        raise SystemExit(0)

    clip_dicts = []
    for p in existing:
        info = probe_clip(p)
        clip_dicts.append({
            "path": str(p),
            "duration_us": info.duration_us,
            "width": info.width,
            "height": info.height,
        })

    test_draft_name = "capcut_reel_bot_smoke_test"
    test_root_meta = Path(config.CAPCUT_DRAFT_ROOT) / config.ROOT_META_FILENAME

    dest = build_draft(test_draft_name, clip_dicts, test_root_meta)
    draft = load_draft(dest)
    print(f"Built draft at {dest}")
    print(f"  segments: {len(draft['tracks'][0]['segments'])}")
    print(f"  videos:   {len(draft['materials']['videos'])}")
    print(f"  duration: {draft['duration']} us")
    assert draft["duration"] == sum(c["duration_us"] for c in clip_dicts)
    print("OK: duration matches sum of clip durations.")
