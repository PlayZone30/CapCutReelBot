# capcut_reel_bot

Automates the first pass of turning raw phone footage into a CapCut rough
cut: pull clips from a Google Drive folder, have Gemini watch them and
segment each into individual camera shots with a quality score, trim the
good shots out, optionally re-time them, and write a real CapCut draft
that opens directly in the app for a human editing pass.

This is a rough-cut assistant, not a finishing tool — every draft it
produces is meant to be opened in CapCut and edited further (trim
adjustments, transitions, color, music, text). Nothing here exports a
final video; it writes a CapCut *draft*.

## How it works (pipeline order)

```
Google Drive folder
  -> drive_ingest.py    download raw clips into ./raw
  -> probe.py            ffprobe: fps, duration, resolution, rotation tag per clip
  -> probe.py            normalize rotation (bake in any rotation tag so every
                         downstream step works on correctly-oriented pixels)
  -> merge_clips.py      bin-pack clips into ~3-minute batches before they ever
                         reach Gemini (keeps a 46-clip shoot from costing 46
                         separate Gemini requests)
  -> gemini_score.py     Gemini watches each batch and returns every distinct
                         camera shot in it — type (pan/push-in/static/etc),
                         in/out range, focus, stability, and a 0-10 score
  -> trim.py             cut each surviving shot out of its batch at Gemini's
                         recommended in/out range
  -> speed_process.py    optionally bake a uniform speed change into the
                         trimmed shots (setpts, or RIFE interpolation if the
                         resulting fps drops too low)
  -> reframe.py          center-crop any shot that's still landscape after
                         rotation-normalize to vertical 1080x1920
  -> draft_writer.py     clone a reference CapCut draft, copy the surviving
                         shots into it, set the canvas to match the footage,
                         and register the draft so it shows up in CapCut
```

Every stage above is also its own standalone script — you can run
`python probe.py <file>` or `python gemini_score.py <file>` directly
without going through the full pipeline. `main.py` is the orchestrator
that chains them together.

### Why this specific order

- **Rotation is fixed before anything else.** Phone footage frequently
  stores a landscape pixel grid (e.g. 1920x1080) plus a rotation tag
  instead of true portrait pixels. Every later step needs to see the
  *displayed* orientation, not the raw grid, so this runs first.
- **Clips are merged into ~3-minute batches before Gemini ever sees them.**
  Gemini is billed and rate-limited per request regardless of how short
  the video is. Sending 46 raw clips (many 5-15s each) as 46 requests
  burns most of a free-tier daily quota on overhead, not analysis.
  Concatenating first means the same footage costs a fraction of the
  requests. See `merge_clips.py`'s docstring for the exact bin-packing
  rule.
- **Trim happens before any speed change, and scoring happens at native
  speed.** Three independent reasons converge on this order: RIFE frame
  interpolation (the expensive part of `speed_process.py`) then only ever
  touches the seconds of footage that actually survive the cut, not the
  whole raw batch; Gemini's own cost scales with video duration, so
  scoring the untrimmed-but-not-yet-slowed clip is the cheapest version
  that still shows real camera movement; and judgments like "is this pan
  steady" are a property of the real capture, not of a stretched-time
  render.

## What actually gets written to CapCut

`draft_writer.py` clones `templates/reference_draft/` (a real, hand-
captured CapCut draft used purely as a schema template — see
`implementationplan.md` for the full field-by-field reverse-engineering
notes) rather than generating a draft from scratch. For every draft it
builds, it also:

- Copies each surviving clip into `<draft_folder>/materials/` and points
  the draft at that copy, rather than referencing the file in place.
  CapCut on macOS is a sandboxed app and can only read files that live
  somewhere it's explicitly been granted access to — its own draft
  folders, or files picked via a native dialog. It cannot read arbitrary
  paths on disk even with Full Disk Access granted; this was confirmed
  hands-on (clips showed "Unsupported Media" until this fix).
- Sets `canvas_config` to match the actual footage's orientation instead
  of inheriting whatever orientation the template happened to be
  captured in (fixes black pillarbox bars when the template and the
  footage disagree on portrait vs landscape).
- Fixes every cross-reference in the cloned draft folder that pointed
  back at the *template's* ids/paths/materials (`timeline_layout.json`,
  `draft_meta_info.json`, `draft_virtual_store.json`, `key_value.json`) —
  the template is a real captured draft with its own history, and none of
  that history belongs in a newly generated draft.
- Registers the draft in `root_meta_info.json` so it actually shows up in
  CapCut's project list, using the *same* draft id that's written inside
  the draft's own metadata (previously these could silently diverge).

## Requirements

- macOS with CapCut installed (Windows draft schema is a different
  filename — `draft_content.json` instead of `draft_info.json` — and is
  unverified; see `config.py`)
- Python 3.11 (this project was built and tested against a `conda`
  environment named `capcut_reel_bot` running 3.11)
- [ffmpeg](https://ffmpeg.org/) with `libx264` support, on your `PATH`
  (`brew install ffmpeg` on macOS)
- A Gemini API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey))
- A Google Cloud service account with the Drive API enabled, for
  downloading from Drive (see setup below — this is the robust path; a
  plain API key or no auth at all both work as fallbacks with caveats,
  see `drive_ingest.py`)

## Setup

### 1. Clone and create the environment

```bash
git clone <this-repo> capcut_reel_bot
cd capcut_reel_bot

conda create -n capcut_reel_bot python=3.11
conda activate capcut_reel_bot
pip install -r requirements.txt
```

### 2. Install ffmpeg

```bash
brew install ffmpeg
```

Confirm it's on your `PATH`:

```bash
ffmpeg -version
ffprobe -version
```

### 3. Get a Gemini API key

Create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
Free tier is enough to test with, but is capped at 5 requests/minute and
20 requests/day — the pipeline's rate limiter (`rate_limiter.py`) enforces
this automatically so you don't blow through it, but it does mean a large
shoot may need more than one day's quota. See `.env.example` for
`GEMINI_RPM`/`GEMINI_RPD` if you're on a paid tier with higher limits.

### 4. Set up Google Drive access (service account, recommended)

The pipeline needs to read whatever Google Drive folder holds your raw
clips. The most reliable way to do this:

1. Go to [console.cloud.google.com/iam-admin/serviceaccounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
   and create a new service account (any project; a free one is fine).
2. Under that service account, go to **Keys → Add key → Create new key →
   JSON**. This downloads a `credentials.json`-style file.
3. Enable the **Google Drive API** for that same Cloud project
   ([console.cloud.google.com/apis/library/drive.googleapis.com](https://console.cloud.google.com/apis/library/drive.googleapis.com)).
4. Save the downloaded key file somewhere in this repo (e.g.
   `credentials.json`) — **do not rename it to something outside the
   `.gitignore` patterns** (`credentials*.json` and `service_account*.json`
   are both already ignored).
5. If your Drive folder is already shared "Anyone with the link", you're
   done — no need to re-share it with the service account's email. That
   sharing setting already grants access to any Google identity,
   including service accounts. If the folder is private, share it
   explicitly with the service account's `...@...iam.gserviceaccount.com`
   email instead.

**Why a service account rather than just an API key:** a bare API key is
treated as anonymous traffic by Google's front-door anti-abuse system,
which is more prone to blocking large/concurrent binary downloads (a
"computer or network may be sending automated queries" 403 — confirmed
hands-on during development, and consistent with multiple independent
reports online). A service account is authenticated, not anonymous, and
doesn't hit this. A plain API key still works as a fallback if you can't
create a service account; see `.env.example`.

### 5. Configure `.env`

```bash
cp .env.example .env
```

Fill in, at minimum:

```ini
GEMINI_API_KEY=your-key-here
DRIVE_FOLDER_ID=the-folder-id-from-the-drive-url
GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/to/credentials.json
```

Everything else in `.env.example` has a documented, working default —
read the comments in that file before changing anything, most of the
"why" is explained inline.

### 6. Verify the setup

```bash
python config.py
```

This prints what the pipeline actually resolved from your `.env` —
confirm `GEMINI_API_KEY set: True`, the right `DRIVE_FOLDER_ID`, and that
`CAPCUT_DRAFT_ROOT` points at your real CapCut projects folder (default
on macOS: `~/Movies/CapCut/User Data/Projects/com.lveditor.draft`).

## Running it

Full pipeline, end to end, from a fresh Drive folder:

```bash
python main.py --draft-name my_reel
```

Open CapCut afterward — the draft named `my_reel` should be in your
project list.

### Useful flags

```bash
python main.py --skip-ingest              # reuse whatever's already in ./raw instead of re-downloading
python main.py --skip-score                # skip Gemini entirely; keep each clip as one full-length shot
python main.py --min-score 6.0              # drop any shot Gemini scored below this (0-10)
python main.py --speed-factor 0.5           # bake this speed into every surviving shot (0.5 = half speed)
python main.py --skip-merge                 # score every clip separately instead of batching first (uses far more Gemini requests)
python main.py --merge-target-seconds 170   # change the ~3min batch-size target before merging
```

### Recommended first run

Before spending real Gemini quota, do a dry run with `--skip-score` to
confirm ingestion, rotation-normalize, merge, and the draft write all
work against your actual footage:

```bash
python main.py --skip-score --draft-name dry_run_test
```

Then run for real once that looks right:

```bash
python main.py --draft-name my_reel
```

### Running individual stages

Every module has its own `if __name__ == "__main__":` entry point for
testing in isolation:

```bash
python probe.py raw/some_clip.MP4              # inspect fps/rotation/orientation
python drive_ingest.py                          # just download, nothing else
python gemini_score.py raw/some_clip.MP4         # score one clip, print the shot breakdown
python trim.py in.mp4 out.mp4 2.0 8.5             # cut seconds 2.0-8.5 out of a file
python speed_process.py in.mp4 out.mp4 0.5        # bake 0.5x speed into a file
python reframe.py in.mp4 out.mp4                  # center-crop to 1080x1920 if landscape
python draft_writer.py                            # smoke-test draft creation with sample clips
python merge_clips.py <output_dir> <clip1> <clip2> ...   # test the bin-packing/merge logic directly
```

## Configuration reference

All of these live in `.env` — see `.env.example` for the full, commented
list. The ones most worth knowing about:

| Variable | Default | What it controls |
|---|---|---|
| `GEMINI_API_KEY` | — | required |
| `GEMINI_MODEL` | `gemini-3.8-flash` | which Gemini model scores clips |
| `GEMINI_RPM` / `GEMINI_RPD` | `5` / `20` | your key's rate limits (free tier defaults) |
| `GEMINI_MAX_WORKERS` | `5` | how many clips to upload/score concurrently |
| `DRIVE_FOLDER_ID` | — | required; the folder ID from the Drive URL |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | — | recommended; path to your service account JSON key |
| `GOOGLE_API_KEY` | — | fallback-only Drive auth; see setup section above |
| `DRIVE_MAX_WORKERS` | `6` | concurrent Drive downloads |
| `FFMPEG_MAX_WORKERS` | `2` | concurrent ffmpeg processes for rotation-normalize/merge (see note below) |
| `CAPCUT_DRAFT_ROOT` | OS default | override if CapCut's project folder isn't in the default location |

### A note on `FFMPEG_MAX_WORKERS`

Rotation-normalize and merge each run their ffmpeg calls concurrently, up
to this many at once. It defaults to a conservative `2` rather than
scaling to your machine's core count, for two reasons: each `ffmpeg`
process using `libx264` is itself internally multi-threaded (so 2
processes can easily use most of an 8-core machine already), and you may
want to keep using the machine for other things while a long batch runs.
Raise it if you know you have CPU/RAM headroom to spare; on an 8GB
machine, going much past 3-4 risks memory pressure from several
simultaneous 1080p60 decode/encode buffers.

**On hardware-accelerated encoding (Apple Silicon):** ffmpeg on macOS
also supports `h264_videotoolbox`, Apple's hardware H.264 encoder.
Measured on the M1 this project was developed on: a single hardware
encode ran about 1.6x faster than the current `libx264` settings, at
less than half the CPU (210% vs 502%) — real headroom if you want it.
However, running multiple concurrent hardware sessions was measured to
provide **no** additional speedup on this chip (2 concurrent sessions
took ~2x as long as one, i.e. they serialize rather than parallelize),
consistent with Apple's media engine having a limited number of
concurrent hardware encode pipelines on base M1-class chips. It's also a
real quality trade-off: `h264_videotoolbox` doesn't support `-crf` (uses
`-q:v` instead, not a direct equivalent) and is widely reported to be
lower quality per bitrate than `libx264` at comparable settings. This
project currently uses `libx264` everywhere for consistent, predictable
quality and genuine multi-process parallelism; switching any stage to
`h264_videotoolbox` would trade some quality for single-job speed, not
for more concurrency, so it hasn't been adopted by default. If you want
to try it, replace `-c:v libx264 -preset veryfast -crf 18` with
`-c:v h264_videotoolbox -q:v 65` (adjust to taste) in the relevant
functions in `probe.py`, `reframe.py`, `merge_clips.py`, `trim.py`, or
`speed_process.py`, and drop `FFMPEG_MAX_WORKERS` to `1` for those
stages since concurrency won't help.

## Known limitations / open items

- **Windows is unverified.** `config.py` branches on OS and assumes
  Windows CapCut/JianYing uses `draft_content.json` instead of
  `draft_info.json`, based on community tooling, not a captured Windows
  draft. Confirm on a real Windows install before relying on it.
- **Mixed-orientation footage in one draft still gets pillarboxed on the
  minority orientation.** `canvas_config` is one setting per draft, not
  per segment — if a draft ends up with both portrait and landscape
  clips, one orientation will always show bars against the other's
  canvas. This project's actual footage has been consistently ~100%
  portrait in practice, so this hasn't needed a real fix, but it's a
  property of how CapCut's timeline works, not something fixable in the
  JSON.
- **RIFE frame interpolation requires a separate binary**
  (`rife-ncnn-vulkan`) not installed by this project's dependencies. If
  `speed_process.py` picks the RIFE branch (effective fps drops below
  ~24 after a speed change) and the binary isn't found, it raises clearly
  rather than silently falling back to a choppier result.
- **Speed changes currently mute audio** is *not* implemented — audio
  currently goes through ffmpeg's `atempo` filter when a speed change is
  applied, which can sound choppy/metallic on non-musical source audio
  (footsteps, ambient noise). This was flagged during development but a
  fix was deliberately held pending a decision on whether muting
  sped-up/slowed shots is the right default — check with whoever owns
  the creative call before changing this.
- **Transitions and filters are not implemented.** Every cut in a
  generated draft is currently a hard cut with no transition and no
  color/style filter applied. This was scoped as a possible next step
  (see project notes) but no code exists for it yet.
