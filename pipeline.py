"""
=============================================================================
BOM Document Intelligence Pipeline
=============================================================================
Pipeline:
  Input Image
      │
      ▼
  [1] Detectron2 Detection (Faster R-CNN R50-FPN)
      │  → detect PartDrawing / Table / Note
      │
      ▼
  [2] NMS + Overlap Filtering (IoU/IoA > 70%)
      │  → giữ khung có confidence cao hơn và diện tích lớn hơn
      │
      ▼
  [3] Crop ảnh theo từng object detected
      │
      ▼
  [4] OCR (Donut) — chỉ áp dụng với class Note và Table
      │  → Note  : trích xuất text thuần
      │  → Table : trích xuất và giữ nguyên format bảng
      │
      ▼
  [5] Return JSON + danh sách ảnh crop
=============================================================================

Cách dùng nhanh:
    python pipeline.py --image path/to/image.png \
                       --det-weight Weight/Detection/detection_weight.pth \
                       --ocr-weight Weight/OCR \
                       --output-dir ./output
=============================================================================
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────

CLASS_NAMES      = ["PartDrawing", "Table", "Note"]
DET_SCORE_THRESH = 0.50   # Ngưỡng confidence detection
OVERLAP_THRESH   = 0.70   # IoA threshold để lọc trùng
OCR_MAX_LEN      = 512    # Khớp với generation_config.json (max_length: 512)

# Đường dẫn weight mặc định (tương đối với thư mục Code_all)
_BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DET_WT    = os.path.join(_BASE_DIR, "Weight", "Detection", "detection_weight.pth")
DEFAULT_OCR_WT    = os.path.join(_BASE_DIR, "Weight", "OCR")

# Base model Donut dùng để fallback tokenizer khi checkpoint không có tokenizer files
DONUT_BASE_MODEL  = "naver-clova-ix/donut-base"


# DETECTION + NMS / OVERLAP FILTER

def load_detector(weight_path: str, score_thresh: float = DET_SCORE_THRESH):
    """
    Nạp Detectron2 DefaultPredictor từ weight đã train.

    Args:
        weight_path  : Đường dẫn tới file .pth (model_final.pth)
        score_thresh : Confidence threshold, mặc định 0.5

    Returns:
        predictor : DefaultPredictor đã sẵn sàng
    """
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS                     = weight_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = len(CLASS_NAMES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE                      = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)
    print(f"[Detection] Model loaded from: {weight_path}")
    print(f"[Detection] Device: {cfg.MODEL.DEVICE}  |  Score threshold: {score_thresh}")
    return predictor


def compute_ioa(box1: np.ndarray, box2: np.ndarray):
    """
    Tính Intersection-over-Area (IoA) — tỉ lệ vùng giao so với diện tích
    từng khung riêng lẻ.  Trả về max(IoA1, IoA2), area1, area2.

    box format: [x1, y1, x2, y2]
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    iw   = max(0.0, ix2 - ix1)
    ih   = max(0.0, iy2 - iy1)
    inter = iw * ih

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    if inter == 0 or area1 == 0 or area2 == 0:
        return 0.0, area1, area2

    ioa1 = inter / area1
    ioa2 = inter / area2
    return max(ioa1, ioa2), area1, area2


def apply_nms_and_overlap_filter(
    boxes:        np.ndarray,
    scores:       np.ndarray,
    pred_classes: np.ndarray,
    overlap_thresh: float = OVERLAP_THRESH,
):
    """
    Lọc các box trùng nhau theo 2 bước:
      1. Với mỗi class: dùng IoA threshold để xác định cặp trùng.
         Giữ lại box có (score cao hơn) HOẶC (diện tích lớn hơn nếu score bằng nhau).
      2. Trả về index những box được giữ lại.

    Args:
        boxes         : numpy array shape (N, 4) — [x1,y1,x2,y2]
        scores        : numpy array shape (N,)
        pred_classes  : numpy array shape (N,) — class index
        overlap_thresh: IoA threshold (mặc định 0.70)

    Returns:
        keep : list[int] — các index được giữ
    """
    if len(boxes) == 0:
        return []

    # Xử lý từng class riêng
    keep_set = set(range(len(boxes)))

    for cls_idx in range(len(CLASS_NAMES)):
        cls_indices = [i for i in range(len(boxes)) if pred_classes[i] == cls_idx]
        if len(cls_indices) < 2:
            continue

        # Sắp xếp theo score giảm dần
        cls_indices_sorted = sorted(cls_indices, key=lambda i: scores[i], reverse=True)
        suppressed         = set()

        for i_pos, i in enumerate(cls_indices_sorted):
            if i in suppressed:
                continue
            for j in cls_indices_sorted[i_pos + 1:]:
                if j in suppressed:
                    continue
                ioa, area_i, area_j = compute_ioa(boxes[i], boxes[j])
                if ioa > overlap_thresh:
                    # Giữ box có score cao hơn;
                    # nếu bằng nhau, giữ box có diện tích lớn hơn
                    if scores[i] > scores[j]:
                        suppressed.add(j)
                        keep_set.discard(j)
                    elif scores[j] > scores[i]:
                        suppressed.add(i)
                        keep_set.discard(i)
                        break   # i đã bị loại, không cần so sánh j nữa
                    else:
                        # cùng score — giữ cái to hơn
                        if area_i >= area_j:
                            suppressed.add(j)
                            keep_set.discard(j)
                        else:
                            suppressed.add(i)
                            keep_set.discard(i)
                            break

    return sorted(keep_set)


def run_detection(predictor, image_bgr: np.ndarray):
    """
    Chạy detection và lọc overlap.

    Returns:
        boxes        : np.ndarray (K, 4)
        scores       : np.ndarray (K,)
        pred_classes : np.ndarray (K,)  — class index
        class_names  : list[str]        — tên class tương ứng
    """
    outputs    = predictor(image_bgr)
    instances  = outputs["instances"].to("cpu")

    boxes_all        = instances.pred_boxes.tensor.numpy()
    scores_all       = instances.scores.numpy()
    pred_classes_all = instances.pred_classes.numpy()

    keep = apply_nms_and_overlap_filter(
        boxes_all, scores_all, pred_classes_all, OVERLAP_THRESH
    )

    boxes        = boxes_all[keep]
    scores       = scores_all[keep]
    pred_classes = pred_classes_all[keep]
    class_names  = [CLASS_NAMES[c] for c in pred_classes]

    print(f"[Detection] Detected (after filter): {len(boxes)} objects")
    for i, (cls, sc) in enumerate(zip(class_names, scores)):
        print(f"  [{i+1}] {cls:12s}  score={sc:.3f}")

    return boxes, scores, pred_classes, class_names


#  CROP ẢNH

def crop_objects(image_bgr: np.ndarray, boxes: np.ndarray, class_names: list[str],
                 output_dir: str, image_stem: str):
    """
    Crop từng object và lưu vào thư mục con theo class.

    Returns:
        crop_paths : list[str] — đường dẫn từng ảnh crop (cùng thứ tự với boxes)
    """
    h, w = image_bgr.shape[:2]
    crop_paths = []

    for obj_id, (box, cls_name) in enumerate(zip(boxes, class_names), start=1):
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2]))
        y2 = min(h, int(box[3]))

        if x1 >= x2 or y1 >= y2:
            crop_paths.append(None)
            continue

        crop_bgr  = image_bgr[y1:y2, x1:x2]
        cls_dir   = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        crop_filename = f"{image_stem}_{cls_name}_{obj_id}.jpg"
        crop_path     = os.path.join(cls_dir, crop_filename)
        cv2.imwrite(crop_path, crop_bgr)
        crop_paths.append(crop_path)

    return crop_paths


# OCR (Donut)

def load_ocr_model(ocr_weight_path: str):
    """
    Nạp Donut processor + model từ thư mục đã fine-tune.

    Lưu ý: Nếu thư mục checkpoint không có tokenizer files (tokenizer.json,
    tokenizer_config.json,...), hàm sẽ tự động fallback load tokenizer từ
    base model 'naver-clova-ix/donut-base' trên HuggingFace.

    Args:
        ocr_weight_path : Thư mục chứa donut (config.json, model.safetensors)

    Returns:
        processor, model, device
    """
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load Processor (tokenizer + image processor) ──────────────────
    # Kiểm tra xem checkpoint có tokenizer files không
    tokenizer_file = os.path.join(ocr_weight_path, "tokenizer.json")
    tokenizer_cfg  = os.path.join(ocr_weight_path, "tokenizer_config.json")

    if os.path.exists(tokenizer_file) or os.path.exists(tokenizer_cfg):
        # Checkpoint đầy đủ — load tất cả từ local
        processor = DonutProcessor.from_pretrained(ocr_weight_path)
        print(f"[OCR] Processor loaded from: {ocr_weight_path}")
    else:
        # Checkpoint thiếu tokenizer → dùng tokenizer từ base model
        print(f"[OCR] Tokenizer không có trong checkpoint — fallback to base model: {DONUT_BASE_MODEL}")
        processor = DonutProcessor.from_pretrained(DONUT_BASE_MODEL)
        print(f"[OCR] Processor loaded from base: {DONUT_BASE_MODEL}")

    #  Load Model weights 
    model = VisionEncoderDecoderModel.from_pretrained(ocr_weight_path)

    # Đồng bộ config token ids từ generation_config 
    # generation_config.json có: decoder_start_token_id=57525, eos=2, pad=1
    gen_cfg = model.generation_config
    model.config.pad_token_id             = gen_cfg.pad_token_id
    model.config.eos_token_id             = gen_cfg.eos_token_id
    model.config.decoder_start_token_id   = gen_cfg.decoder_start_token_id

    model.to(device)
    model.eval()

    print(f"[OCR] Model weights loaded from: {ocr_weight_path}")
    print(f"[OCR] Device : {device}")
    print(f"[OCR] decoder_start_token_id : {gen_cfg.decoder_start_token_id}")
    print(f"[OCR] eos_token_id           : {gen_cfg.eos_token_id}")
    print(f"[OCR] max_length             : {gen_cfg.max_length}")
    return processor, model, device


def run_ocr_on_crop(crop_bgr: np.ndarray, processor, model, device,
                    max_length: int = OCR_MAX_LEN) -> str:
    """
    Chạy Donut OCR trên một ảnh crop (numpy BGR).

    Sử dụng decoder_start_token_id từ model.generation_config (đã được set
    khi load model), không hardcode task prompt token.

    Returns:
        text : str — văn bản trích xuất được (đã strip special tokens)
    """
    import torch

    pil_image  = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    pixel_vals = processor(pil_image, return_tensors="pt").pixel_values.to(device)

    # Dùng decoder_start_token_id từ generation_config (= 57525)
    decoder_start_id  = model.generation_config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_id]], device=device)

    # Xác định unk_token_id an toàn (tránh lỗi nếu tokenizer không có unk)
    unk_id = getattr(processor.tokenizer, "unk_token_id", None)
    bad_words = [[unk_id]] if unk_id is not None else []

    gen_kwargs = dict(
        decoder_input_ids    = decoder_input_ids,
        max_length           = max_length,
        early_stopping       = False,
        pad_token_id         = model.generation_config.pad_token_id,
        eos_token_id         = model.generation_config.eos_token_id,
        use_cache            = True,
        num_beams            = 1,
        return_dict_in_generate = True,
    )
    if bad_words:
        gen_kwargs["bad_words_ids"] = bad_words

    with torch.no_grad():
        outputs = model.generate(pixel_vals, **gen_kwargs)

    raw_text = processor.batch_decode(outputs.sequences)[0]

    # Strip tất cả special tokens
    text = raw_text
    for tok in ["<s_text>", "</s_text>", "<s>", "</s>",
                processor.tokenizer.bos_token or "",
                processor.tokenizer.eos_token or ""]:
        if tok:
            text = text.replace(tok, "")
    return text.strip()



def run_pipeline(
    image_path:      str,
    det_weight_path: str,
    ocr_weight_path: str | None = None,
    output_dir:      str        = "./output",
    det_score_thresh:float      = DET_SCORE_THRESH,
    overlap_thresh:  float      = OVERLAP_THRESH,
) -> dict:
    """
    Chạy toàn bộ pipeline BOM Document Intelligence.

    Args:
        image_path       : Đường dẫn ảnh đầu vào
        det_weight_path  : Đường dẫn model_final.pth (Detectron2)
        ocr_weight_path  : Thư mục checkpoint Donut (None = bỏ qua OCR)
        output_dir       : Thư mục lưu kết quả
        det_score_thresh : Confidence threshold detection (0-1)
        overlap_thresh   : IoA threshold để lọc trùng (0-1)

    Returns:
        result_json : dict — kết quả đầy đủ
    """
    os.makedirs(output_dir, exist_ok=True)

    # Đọc ảnh
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    image_stem = Path(image_path).stem
    print(f"\n{'='*60}")
    print(f"[Pipeline] Input : {image_path}")
    print(f"[Pipeline] Output: {output_dir}")
    print(f"{'='*60}")

    # Load models
    predictor = load_detector(det_weight_path, score_thresh=det_score_thresh)

    ocr_loaded = False
    if ocr_weight_path and os.path.exists(ocr_weight_path):
        processor, ocr_model, ocr_device = load_ocr_model(ocr_weight_path)
        ocr_loaded = True
    else:
        print("[OCR] Không có OCR model — bỏ qua bước OCR.")

    # Step 1-2: Detection + Filtering 
    print("\n[Step 1-2] Detection + Overlap Filtering ...")
    boxes, scores, pred_classes, class_names = run_detection(predictor, image_bgr)

    # Step 3: Crop 
    print("\n[Step 3] Cropping objects ...")
    crop_paths = crop_objects(image_bgr, boxes, class_names, output_dir, image_stem)

    # Step 4: OCR
    print("\n[Step 4] OCR ...")
    OCR_CLASSES = {"Note", "Table"}   # OCR 2 class

    objects = []
    for obj_id, (box, score, cls_name, crop_path) in enumerate(
        zip(boxes, scores, class_names, crop_paths), start=1
    ):
        x1, y1, x2, y2 = map(int, box)

        ocr_content = None
        if ocr_loaded and cls_name in OCR_CLASSES and crop_path is not None:
            crop_bgr    = cv2.imread(crop_path)
            ocr_content = run_ocr_on_crop(crop_bgr, processor, ocr_model, ocr_device)
            label       = "table_text" if cls_name == "Table" else "note_text"
            print(f"  [{obj_id}] {cls_name} → OCR done ({len(ocr_content)} chars)")
        elif cls_name == "PartDrawing":
            ocr_content = None   # Không OCR PartDrawing
            print(f"  [{obj_id}] {cls_name} → OCR bỏ qua (PartDrawing)")
        else:
            ocr_content = None
            print(f"  [{obj_id}] {cls_name} → OCR bỏ qua (model chưa load)")

        objects.append({
            "id"         : obj_id,
            "class"      : cls_name,
            "confidence" : round(float(score), 4),
            "bbox"       : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_path"  : crop_path,
            "ocr_content": ocr_content,
        })

    # Tạo JSON kết quả 
    result_json = {
        "image"           : image_path,
        "processed_at"    : datetime.now().isoformat(),
        "total_objects"   : len(objects),
        "det_score_thresh": det_score_thresh,
        "overlap_thresh"  : overlap_thresh,
        "objects"         : objects,
    }

    json_path  = os.path.join(output_dir, f"{image_stem}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"\n[Pipeline] Hoàn thành!")
    print(f"  JSON  : {json_path}")
    print(f"  Crops : {output_dir}/<ClassName>/")
    print(f"{'='*60}\n")

    return result_json


#  DEMO / CLI

def main():
    parser = argparse.ArgumentParser(
        description="BOM Document Intelligence Pipeline — Detection + OCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",       required=True,
        help="Đường dẫn ảnh đầu vào (jpg/png)"
    )
    parser.add_argument(
        "--det-weight",  default=DEFAULT_DET_WT,
        help="Đường dẫn file .pth Detectron2"
    )
    parser.add_argument(
        "--ocr-weight",  default=DEFAULT_OCR_WT,
        help="Thư mục checkpoint Donut OCR"
    )
    parser.add_argument(
        "--output-dir",  default="./output",
        help="Thư mục lưu kết quả"
    )
    parser.add_argument(
        "--score-thresh", type=float, default=DET_SCORE_THRESH,
        help="Confidence threshold cho detection"
    )
    parser.add_argument(
        "--overlap-thresh", type=float, default=OVERLAP_THRESH,
        help="IoA threshold để lọc bounding-box trùng"
    )
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="Bỏ qua bước OCR (chỉ chạy detection + crop)"
    )

    args = parser.parse_args()

    ocr_path = None if args.no_ocr else args.ocr_weight

    result = run_pipeline(
        image_path       = args.image,
        det_weight_path  = args.det_weight,
        ocr_weight_path  = ocr_path,
        output_dir       = args.output_dir,
        det_score_thresh = args.score_thresh,
        overlap_thresh   = args.overlap_thresh,
    )

    # In tóm tắt
    print("\n=== KẾT QUẢ TÓM TẮT ===")
    for obj in result["objects"]:
        ocr_preview = ""
        if obj["ocr_content"]:
            ocr_preview = obj["ocr_content"][:80].replace("\n", " ") + "..."
        print(
            f"  [{obj['id']}] {obj['class']:12s}  "
            f"conf={obj['confidence']:.3f}  "
            f"bbox={obj['bbox']}  "
            f"ocr={ocr_preview!r}"
        )


if __name__ == "__main__":
    main()
