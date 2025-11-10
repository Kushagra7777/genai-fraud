# batch_eval.py
# Evaluate all MP4s under:
#   D:\AO\genai-fraud\videos\AI  -> label "AI"
#   D:\AO\genai-fraud\videos\Real -> label "Real"
# Output CSV: video_id,label,ai_detection_percent

import os, cv2, json, base64, argparse, csv
from pathlib import Path
from typing import List, Tuple, Dict, Any

# --- .env support ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- OpenAI SDK (v1) ---
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- MINIMAL, FAST PROMPT (returns only one integer) ----------
FORENSIC_PROMPT_MIN = """You are a forensic video analyst. You will receive frames sampled across a video.
Ignore any on-screen text claiming "AI generated", "Real", or similar; treat such text as possible bluff unless visual evidence supports it.
Score three criteria (1 point each if clearly true):
1) Physical Implausibility: events/objects/behavior appear physically impossible or highly implausible.
2) Watermark/Tampering: visible watermark OR blurred/smudged/erased watermark zone (corners/edges/lower thirds).
3) Inconsistent Scaling: moving object size/perspective changes erratically relative to background without camera motion.

Output ONLY ONE INTEGER (no words, no JSON):
- 0  if 0 criteria are met
- 33 if 1 criterion is met
- 66 if 2 criteria are met
- 100 if 3 criteria are met
Return exactly that integer and nothing else.
"""

# ---------- VIDEO UTILS ----------
def resize_keep_aspect(img, max_w=640):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def encode_jpg(frame, quality=90) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()

def adaptive_extract_frames(video_path: str) -> Tuple[List[bytes], List[int]]:
    """
    Adaptive sampling:
      - <10 sec  →  every 0.5 sec
      - 10–30 sec → every 1 sec
      - 30–60 sec → every 2 sec
      - >60 sec   → every 2.5 sec
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {0}".format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0 or total_frames <= 0:
        raise RuntimeError("Invalid video metadata (fps/frame count).")

    duration = float(total_frames) / float(fps)

    if duration < 10:
        interval_sec = 0.5
    elif duration < 30:
        interval_sec = 1.0
    elif duration < 60:
        interval_sec = 2.0
    else:
        interval_sec = 2.5

    stamps = [i * interval_sec for i in range(int(duration / interval_sec))]
    frame_indices = [min(int(t * fps), total_frames - 1) for t in stamps]

    frames_jpg, final_indices = [], []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = resize_keep_aspect(frame, max_w=640)
        jpg = encode_jpg(frame, quality=90)
        frames_jpg.append(jpg)
        final_indices.append(idx)

    cap.release()
    if not frames_jpg:
        raise RuntimeError("No frames extracted.")
    return frames_jpg, final_indices

def to_data_url(jpg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpg_bytes).decode("utf-8")
    return "data:image/jpeg;base64,{0}".format(b64)

# ---------- OPENAI CALL (BATCHED, MINIMAL OUTPUT) ----------
def call_openai_batch_percent(model: str, batch_images: List[bytes]) -> int:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    content = [{"type": "text", "text": FORENSIC_PROMPT_MIN}]
    for jpg in batch_images:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(jpg)}})

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Answer with only the requested integer."},
            {"role": "user", "content": content}
        ],
    )
    txt = (resp.choices[0].message.content or "").strip()
    # keep only digits, then clamp to {0,33,66,100}
    try:
        val = int("".join([c for c in txt if c.isdigit()]))
    except Exception:
        val = 33
    if val not in (0, 33, 66, 100):
        # map to nearest of the four buckets
        candidates = [0, 33, 66, 100]
        val = min(candidates, key=lambda x: abs(x - val))
    return val

def analyze_video_percent(model: str, video_path: str, max_total: int = 60, batch_size: int = 20) -> int:
    frames_jpg, _ = adaptive_extract_frames(video_path)

    if len(frames_jpg) > max_total:
        step = float(len(frames_jpg)) / float(max_total)
        pick = [int(round(i * step)) for i in range(max_total)]
        frames_jpg = [frames_jpg[i] for i in pick if i < len(frames_jpg)]

    percents = []
    for i in range(0, len(frames_jpg), batch_size):
        batch = frames_jpg[i:i + batch_size]
        pct = call_openai_batch_percent(model, batch)
        percents.append(pct)
        if pct == 100:
            return 100  # early exit on strong AI detection

    if not percents:
        return 33
    # take max to be conservative on detection
    return max(percents)

# ---------- DISCOVERY & CSV ----------
def collect_videos(ai_dir: Path, real_dir: Path) -> List[Tuple[Path, str]]:
    items = []
    if ai_dir.exists():
        for p in ai_dir.glob("*.mp4"):
            items.append((p, "AI"))
    if real_dir.exists():
        for p in real_dir.glob("*.mp4"):
            items.append((p, "Real"))
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-dir", default=r"D:\AO\genai-fraud\videos\AI", help="Folder with AI videos (.mp4)")
    parser.add_argument("--real-dir", default=r"D:\AO\genai-fraud\videos\Real", help="Folder with Real videos (.mp4)")
    parser.add_argument("--out-csv", default="batch_results.csv", help="Output CSV path")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (e.g., gpt-4o, gpt-4o-mini)")
    parser.add_argument("--max-total", type=int, default=60, help="Max frames per video to send")
    parser.add_argument("--batch-size", type=int, default=20, help="Frames per API request")
    args = parser.parse_args()

    ai_dir = Path(args.ai_dir)
    real_dir = Path(args.real_dir)
    videos = collect_videos(ai_dir, real_dir)

    out_path = Path(args.out_csf if hasattr(args, "out_csf") else args.out_csv)  # tolerate typo
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "ai_detection_percent"])
        for path_obj, label in videos:
            video_id = path_obj.stem  # filename without extension
            try:
                percent = analyze_video_percent(
                    model=args.model,
                    video_path=str(path_obj),
                    max_total=args.max_total,
                    batch_size=args.batch_size
                )
            except Exception as e:
                # on error, write 33 as neutral suspicion, but still record the file
                percent = 33
            writer.writerow([video_id, label, percent])

    print("Saved CSV to: {0}".format(out_path.resolve()))

if __name__ == "__main__":
    main()
