from ultralytics import YOLO
import cv2
import os
import json
import argparse
import sys
import time
import csv

# 1️⃣ Load your trained model
model = YOLO("best.pt")  # change to "last.pt" if you want

# 2️⃣ CLI arguments
parser = argparse.ArgumentParser(description="Hand glove detection with logging")
parser.add_argument("--input", default="input_images", help="Input images folder")
parser.add_argument("--output", default="output", help="Output folder for annotated images")
parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0-1)")
parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (e.g., 640)")
parser.add_argument("--device", default=None, help="Device to run on, e.g., '0' for GPU, 'cpu' for CPU")
parser.add_argument("--batch", type=int, default=0, help="Batch size for list inference (0 or 1 = no batching)")
parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers (if supported)")
parser.add_argument("--classes", default=None, help="Comma-separated class names to filter (e.g., 'gloved_hand,bare_hand')")
parser.add_argument("--skip-existing", action="store_true", help="Skip images that already have outputs and logs")
parser.add_argument("--no-annotate", action="store_true", help="Do not save annotated images")
parser.add_argument("--log-path", default="logs.json", help="Path to logs JSON file")
parser.add_argument("--round", type=int, default=None, help="Round floats to N decimals in logs (e.g., 3)")
parser.add_argument("--benchmark", action="store_true", help="Print timing and FPS statistics")
parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
parser.add_argument("--max-det", type=int, default=300, help="Maximum number of detections per image")
parser.add_argument("--half", action="store_true", help="Use half precision (FP16) where supported")
parser.add_argument("--csv-path", default=None, help="Optional path to write per-detection CSV")
args = parser.parse_args()

# Input and output folders
input_path = args.input
output_folder = args.output

os.makedirs(output_folder, exist_ok=True)

logs_path = args.log_path
if os.path.exists(logs_path):
    with open(logs_path, "r", encoding="utf-8") as f:
        try:
            existing = json.load(f)
            if isinstance(existing, dict):
                logs = list(existing.values())
            elif isinstance(existing, list):
                logs = existing
            else:
                logs = []
        except json.JSONDecodeError:
            logs = []
else:
    logs = []

# Helpers
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def list_images(path):
    if os.path.isfile(path):
        return [path] if path.lower().endswith(IMG_EXTS) else []
    files = []
    for root, _, fnames in os.walk(path):
        for fn in fnames:
            if fn.lower().endswith(IMG_EXTS):
                files.append(os.path.join(root, fn))
    return sorted(files)

def chunks(seq, n):
    if n is None or n <= 1:
        yield seq
    else:
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

# 3️⃣ Collect inputs
all_images = list_images(input_path)

if not all_images:
    print("No images found to process. Provide a valid image file or a folder containing images.")
    sys.exit(0)

# 4️⃣ Inference loop (batched if requested)
processed = 0
start_time = time.time()
for batch_paths in chunks(all_images, args.batch if args.batch and args.batch > 1 else None):
    # Optional skip-existing filtering per batch
    if args.skip_existing:
        filtered = []
        for p in batch_paths:
            name = os.path.basename(p)
            out_img = os.path.join(output_folder, name)
            has_log = any((isinstance(it, dict) and it.get("filename") == name) for it in logs)
            if not (os.path.exists(out_img) and has_log):
                filtered.append(p)
        batch_paths = filtered
        if not batch_paths:
            continue
    # Run inference
    try:
        # Map class name filters to indices if provided
        classes_arg = None
        if args.classes:
            name_to_idx = {str(v): k for k, v in model.names.items()}
            classes_arg = []
            for nm in [s.strip() for s in args.classes.split(',') if s.strip()]:
                if nm in name_to_idx:
                    classes_arg.append(name_to_idx[nm])
        results = model.predict(
            source=batch_paths,
            conf=args.confidence,
            save=False,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            classes=classes_arg,
            iou=args.iou,
            max_det=args.max_det,
            half=args.half
        )
    except Exception as e:
        print(f"Prediction failed for batch starting with {os.path.basename(batch_paths[0])}: {e}")
        continue

    # results is a list aligned with batch_paths
    for img_path, res in zip(batch_paths, results):
        img_name = os.path.basename(img_path)

        if not args.no_annotate:
            annotated_img = res.plot()
            try:
                cv2.imwrite(os.path.join(output_folder, img_name), annotated_img)
            except Exception as e:
                print(f"Failed to save annotated image {img_name}: {e}")

        # Print detections
        detections = []
        for box in res.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            bbox = [float(v) for v in box.xyxy[0].tolist()]
            if args.round is not None and args.round >= 0:
                conf = round(conf, args.round)
                bbox = [round(v, args.round) for v in bbox]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": bbox
            })

        record = {"filename": img_name, "detections": detections}

        updated = False
        for i, item in enumerate(logs):
            if isinstance(item, dict) and item.get("filename") == img_name:
                logs[i] = record
                updated = True
                break
        if not updated:
            logs.append(record)

        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

        if args.csv_path:
            write_header = not os.path.exists(args.csv_path)
            try:
                with open(args.csv_path, 'a', newline='', encoding='utf-8') as cf:
                    writer = csv.writer(cf)
                    if write_header:
                        writer.writerow(["filename", "label", "confidence", "x1", "y1", "x2", "y2"])
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        writer.writerow([img_name, det["label"], det["confidence"], x1, y1, x2, y2])
            except Exception as e:
                print(f"Failed to write CSV for {img_name}: {e}")

        processed += 1
        if args.benchmark:
            elapsed = time.time() - start_time
            fps = processed / elapsed if elapsed > 0 else 0.0
            print(f"Processed {processed}/{len(all_images)}: {img_name} | {elapsed:.2f}s elapsed, {fps:.2f} FPS")
        else:
            print(f"Processed {processed}/{len(all_images)}: {img_name}")
        for det in detections:
            print(f" → {det['label']} ({det['confidence']:.2f})")

print("\n Detection completed. Check the 'output' folder.")

