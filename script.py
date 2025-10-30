import os, cv2, json, base64, argparse, math
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

# ---------- PROMPT ----------
FORENSIC_PROMPT = """You are a forensic media analyst. Determine whether these frames sampled every 0.5s across a video suggest the overall video is REAL or AI-GENERATED.

STRICT RULES (override everything):
1) If a visible watermark is present (any kind), the video is 100% AI-GENERATED.
2) If there is obvious blur/smudge/erasure where a watermark would typically be (corners/edges/lower thirds), classify as 100% AI-GENERATED.

Otherwise analyze:
- Temporal coherence between nearby frames (lighting/tone/skin hue consistency).
- Realistic motion blur/rolling shutter; consistent grain/compression signatures.
- Geometry/edges: hands, eyes, teeth, hair, text, reflections/shadows.
- Too-clean microtextures, shimmer/flicker, diffusion-seams, texture crawling.

OUTPUT STRICTLY JSON:
{
  "verdict": "real" | "ai_suspected" | "ai_detected" | "inconclusive",
  "confidence": 0-100,
  "key_signals": [{"name": "...", "evidence": "..."}],
  "per_frame_notes": {"frame_index_<i>": "short note"},
  "overall_rationale": "2-5 sentences"
}

If rule (1) or (2) triggers, respond:
{
  "verdict": "ai_detected",
  "confidence": 100,
  "key_signals": [{"name": "Watermark rule", "evidence": "Watermark or blurred watermark region detected"}],
  "per_frame_notes": {},
  "overall_rationale": "Watermark or its erasure detected; video classified as AI-generated with full confidence."
}
Be conservative; if evidence is weak, use "inconclusive".
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

def extract_interval_frames(video_path: str, interval_sec: float = 0.5) -> Tuple[List[bytes], List[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {0}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cv2.CAP_PROP_FRAME_COUNT and cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0 or total_frames <= 0:
        raise RuntimeError("Video metadata invalid (fps/frame count).")
    duration = float(total_frames) / float(fps)
    frame_indices = [min(int(t * fps), total_frames - 1) for t in [i * interval_sec for i in range(int(duration / interval_sec))]]

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

# ---------- OPENAI CALLS (BATCHED) ----------
def call_openai_batch(model: str, batch_images: List[bytes]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    content = [{"type": "text", "text": FORENSIC_PROMPT}]
    for jpg in batch_images:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(jpg)}})

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are meticulous, cautious, and follow rules precisely."},
            {"role": "user", "content": content}
        ],
    )
    txt = resp.choices[0].message.content.strip()
    return json.loads(txt)

def analyze_batched(model: str, frames: List[bytes], batch_size: int = 20) -> List[Dict[str, Any]]:
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        res = call_openai_batch(model, batch)
        results.append(res)
        if res.get("verdict") == "ai_detected" and int(res.get("confidence", 0)) == 100:
            break
    return results

def merge_results(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch_results:
        return {"verdict": "inconclusive", "confidence": 0, "overall_rationale": "No results."}

    for r in batch_results:
        if r.get("verdict") == "ai_detected" and int(r.get("confidence", 0)) == 100:
            return {
                "verdict": "ai_detected",
                "confidence": 100,
                "overall_rationale": "Watermark rule triggered in at least one batch.",
                "_batches": batch_results
            }

    counts = {"ai_detected": 0, "ai_suspected": 0, "real": 0, "inconclusive": 0}
    avg_conf = []
    for r in batch_results:
        v = r.get("verdict", "inconclusive")
        counts[v] = counts.get(v, 0) + 1
        if isinstance(r.get("confidence"), (int, float)):
            avg_conf.append(float(r["confidence"]))

    if counts["ai_detected"] > 0:
        return {
            "verdict": "ai_detected",
            "confidence": (max(avg_conf) if avg_conf else 85),
            "overall_rationale": "One or more batches detected strong AI signals.",
            "_batches": batch_results
        }
    if counts["ai_suspected"] > 0:
        return {
            "verdict": "ai_suspected",
            "confidence": (sum(avg_conf) / len(avg_conf) if avg_conf else 70),
            "overall_rationale": "Suspicious signals present across batches.",
            "_batches": batch_results
        }
    if counts["real"] > 0 and counts["ai_suspected"] == 0 and counts["ai_detected"] == 0:
        return {
            "verdict": "real",
            "confidence": (sum(avg_conf) / len(avg_conf) if avg_conf else 80),
            "overall_rationale": "No decisive synthetic cues were found in sampled frames.",
            "_batches": batch_results
        }
    return {
        "verdict": "inconclusive",
        "confidence": (sum(avg_conf) / len(avg_conf) if avg_conf else 50),
        "overall_rationale": "Evidence insufficient.",
        "_batches": batch_results
    }

# ---------- I/O HELPERS ----------
def save_frames(folder: Path, frames_jpg: List[bytes], frame_indices: List[int]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(zip(frames_jpg, frame_indices), 1):
        jpg, idx = item
        filename = "t{0:04d}_frame_{1}.jpg".format(i, idx)
        (folder / filename).write_bytes(jpg)

def pretty_console(verdict: str, conf: float):
    v = verdict.lower()
    if v == "ai_detected":
        print("\n\033[91m AI DETECTED ({0}%)\033[0m\n".format(int(conf)))
    elif v == "ai_suspected":
        print("\n\033[93m  AI SUSPECTED ({0}%)\033[0m\n".format(int(conf)))
    elif v == "real":
        print("\n\033[92m REAL ({0}%)\033[0m\n".format(int(conf)))
    else:
        print("\n\033[90m INCONCLUSIVE ({0}%)\033[0m\n".format(int(conf)))

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="video_analysis.json", help="Where to save final JSON")
    ap.add_argument("--save-frames", default=None, help="Optional folder to write extracted frames")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model (e.g., gpt-4o, gpt-4o-mini)")
    ap.add_argument("--interval", type=float, default=0.5, help="Seconds between frames (default 0.5)")
    ap.add_argument("--max-total", type=int, default=60, help="Max total frames to send (to avoid context errors)")
    ap.add_argument("--batch-size", type=int, default=20, help="Images per API request")
    args = ap.parse_args()

    frames_jpg, frame_indices = extract_interval_frames(args.video, interval_sec=args.interval)

    if len(frames_jpg) > args.max_total:
        step = float(len(frames_jpg)) / float(args.max_total)
        pick = [int(round(i * step)) for i in range(args.max_total)]
        frames_jpg = [frames_jpg[i] for i in pick if i < len(frames_jpg)]
        frame_indices = [frame_indices[i] for i in pick if i < len(frame_indices)]

    if args.save_frames:
        save_frames(Path(args.save_frames), frames_jpg, frame_indices)

    batch_results = analyze_batched(args.model, frames_jpg, batch_size=args.batch_size)
    final = merge_results(batch_results)
    final["_frame_indices"] = frame_indices
    Path(args.out).write_text(json.dumps(final, indent=2), encoding="utf-8")

    pretty_console(final.get("verdict", "inconclusive"), float(final.get("confidence", 0)))
    print("Saved analysis to: {0}".format(Path(args.out).resolve()))

if __name__ == "__main__":
    main()
