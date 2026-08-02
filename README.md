---
title: Litter Monitoring Dashboard
emoji: 🚯
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
---

# Litter Monitoring Dashboard

Detects littering events in video (YOLOv8), logs them anonymously (location,
timestamp, object type, blurred-face snapshot), and shows a hotspot dashboard.
No face ID, no automated fines.

## Setup

1. Copy `best.pt` (YOLOv8 litter-detection weights) into this folder.
   Get it from https://github.com/ananya868/Anti-Littering-System-Computer-Vision
   or set `LITTER_MODEL_PATH` to a Hub-hosted copy you download at startup.
2. (Optional) Add Supabase persistence:
   - Create a free project at supabase.com
   - Create table `litter_events` (see schema below)
   - Create a public storage bucket named `evidence`
   - In Space Settings → Variables and secrets, add `SUPABASE_URL` and
     `SUPABASE_KEY` as **secrets**
   - Without these, events are logged to a local `events.jsonl` file instead
     (fine for testing; not durable on an ephemeral Space).
3. Push to a Hugging Face Space with SDK "Gradio", hardware "CPU basic" (free).

```sql
create table litter_events (
    id bigint generated always as identity primary key,
    location_tag text not null,
    detected_at timestamptz default now(),
    object_label text not null,
    confidence float not null,
    evidence_image_url text
);
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LITTER_MODEL_PATH` | `best.pt` | Path to YOLO weights |
| `SUPABASE_URL` / `SUPABASE_KEY` | unset | Enables cloud persistence |
| `SIMPLE_MODE` | `true` | Skip MoveNet hand-tracking, flag every detection as an event |
| `FRAME_SKIP` | `3` | Process every Nth frame |
| `RESIZE_WIDTH` | `640` | Downscale width before inference |

## Local testing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Then open the local URL Gradio prints. Works even without `best.pt` or Supabase
configured — you'll just see a warning banner and an empty detector.
