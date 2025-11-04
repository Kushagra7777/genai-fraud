import os, cv2, json, base64, argparse
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

# ---------- PROMPT (1 frame per second, scoring = 0/33/66/100) ----------
# FORENSIC_PROMPT = """You are a forensic media analyst. Analyze the sequence of frames — sampled once per second from a video — to assess whether the video is REAL or AI-GENERATED.

# Tasks:
# 1) Reconstruct a short narrative from the frames in chronological order (what seems to be happening).
# 2) Judge realism (physics, logic, feasibility).
# 3) Scoring (add 1 point per triggered criterion):
#    - Physical Implausibility: events/objects/behaviors are impossible or highly implausible.
#    - Watermark or Tampering: visible watermark OR suspicious blur/smudging/erasure in typical watermark zones (corners, edges, lower thirds).
#    - Inconsistent Object Scaling: moving object size/perspective shifts erratically relative to background without camera motion.

# Final classification:
# - Total points = 0 → verdict = "real" (ai_probability_percent = 0)
# - Total points = 1 → verdict = "ai_suspected" (ai_probability_percent = 33)
# - Total points = 2 → verdict = "ai_suspected" (ai_probability_percent = 66)
# - Total points = 3 → verdict = "ai_detected" (ai_probability_percent = 100)

# Output strictly as JSON:
# {
#   "verdict": "real" | "ai_suspected" | "ai_detected",
#   "ai_probability_percent": 0 | 33 | 66 | 100,
#   "story_reconstruction": "Brief narrative of what happens across frames",
#   "score_breakdown": {
#     "physical_implausibility": 0 | 1,
#     "watermark_or_tampering": 0 | 1,
#     "inconsistent_object_scaling": 0 | 1
#   },
#   "key_signals": [{"criterion": "...", "observation": "..."}],
#   "per_frame_notes": {"frame_index_<i>": "concise anomaly or observation"},
#   "overall_rationale": "2-5 sentences explaining the score, story realism, and forensic concerns."
# }
# Be conservative; only assign points when evidence is clear.
# """

FORENSIC_PROMPT = """You are a forensic video analyst. Evaluate the sequence of frames (1 per second) and decide if the video is REAL or AI-GENERATED.

Your job:
1. Write one short sentence (max 15 words) describing what the frames show overall.
2. Score three quick criteria (1 point each if clearly true):
   - Physical Implausibility: something looks physically impossible or unnatural.
   - Watermark/Tampering: visible watermark or blurred/erased watermark zone.
   - Inconsistent Scaling: objects change size or perspective unrealistically.

Final classification:
- 0 points → verdict = "real" (ai_probability_percent = 0)
- 1 point → verdict = "ai_suspected" (33)
- 2 points → verdict = "ai_suspected" (66)
- 3 points → verdict = "ai_detected" (100)

Return concise JSON only:
{
  "verdict": "real" | "ai_suspected" | "ai_detected",
  "ai_probability_percent": 0 | 33 | 66 | 100,
  "story": "one line summary of what happens",
  "score": {
    "physical_implausibility": 0 | 1,
    "watermark_or_tampering": 0 | 1,
    "inconsistent_scaling": 0 | 1
  },
  "reasoning": "one line justification combining all criteria"
}

Keep sentences very short and clear. No extra text outside JSON.
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

def extract_interval_frames(video_path: str, interval_sec: float = 1.0) -> Tuple[List[bytes], List[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {0}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cv2.CAP_PROP_FRAME_COUNT and cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0 or total_frames <= 0:
        raise RuntimeError("Video metadata invalid (fps/frame count).")
    duration = float(total_frames) / float(fps)

    # build timestamps 0,1,2,... up to duration (exclusive)
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
        # early exit if any batch reports definitive AI (100)
        if res.get("verdict") == "ai_detected" and int(res.get("ai_probability_percent", 0)) == 100:
            break
    return results

def merge_results(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch_results:
        # no batches → return a cautious result using the same schema
        return {
            "verdict": "ai_suspected",
            "ai_probability_percent": 33,
            "overall_rationale": "No batch results available; defaulting to cautious suspicion."
        }

    # hard priority: any batch with ai_detected and 100% wins
    for r in batch_results:
        if r.get("verdict") == "ai_detected" and int(r.get("ai_probability_percent", 0)) == 100:
            return {
                "verdict": "ai_detected",
                "ai_probability_percent": 100,
                "overall_rationale": "At least one batch indicated definitive AI signals (scored 3/3).",
                "_batches": batch_results
            }

    counts = {"ai_detected": 0, "ai_suspected": 0, "real": 0}
    probs = []
    for r in batch_results:
        v = r.get("verdict", "ai_suspected")
        counts[v] = counts.get(v, 0) + 1
        if isinstance(r.get("ai_probability_percent"), (int, float)):
            probs.append(float(r["ai_probability_percent"]))

    # choose verdict by priority, then average probability as a rough final %
    avg_prob = (sum(probs) / len(probs)) if probs else 33.0

    if counts["ai_detected"] > 0:
        return {
            "verdict": "ai_detected",
            "ai_probability_percent": int(round(max(probs) if probs else 66)),
            "overall_rationale": "Multiple frames/batches show strong AI indicators.",
            "_batches": batch_results
        }
    if counts["ai_suspected"] > 0:
        return {
            "verdict": "ai_suspected",
            "ai_probability_percent": int(round(avg_prob)),
            "overall_rationale": "Some suspicious cues present, but not definitive across all batches.",
            "_batches": batch_results
        }
    # otherwise, lean real
    return {
        "verdict": "real",
        "ai_probability_percent": int(round(avg_prob)) if avg_prob < 50 else 0,
        "overall_rationale": "No decisive synthetic cues in sampled frames.",
        "_batches": batch_results
    }

# ---------- I/O HELPERS ----------
def save_frames(folder: Path, frames_jpg: List[bytes], frame_indices: List[int]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(zip(frames_jpg, frame_indices), 1):
        jpg, idx = item
        name = "t{0:04d}_frame_{1}.jpg".format(i, idx)
        (folder / name).write_bytes(jpg)

def pretty_console(verdict: str, ai_prob: float):
    v = (verdict or "").lower()
    p = int(ai_prob)
    if v == "ai_detected":
        print("\n\033[91m AI DETECTED ({0}%)\033[0m\n".format(p))
    elif v == "ai_suspected":
        print("\n\033[93m  AI SUSPECTED ({0}%)\033[0m\n".format(p))
    elif v == "real":
        print("\n\033[92m REAL ({0}%)\033[0m\n".format(p))
    else:
        print("\n\033[90m RESULT ({0}%)\033[0m\n".format(p))

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="video_analysis.json", help="Where to save final JSON")
    ap.add_argument("--save-frames", default=None, help="Optional folder to write extracted frames")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model (e.g., gpt-4o, gpt-4o-mini)")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between frames (default 1.0 sec)")
    ap.add_argument("--max-total", type=int, default=60, help="Max frames to send (avoid context errors)")
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

    pretty_console(final.get("verdict", "ai_suspected"), float(final.get("ai_probability_percent", 33)))
    print("Saved analysis to: {0}".format(Path(args.out).resolve()))

if __name__ == "__main__":
    main()
