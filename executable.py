"""
IDFC GenAI Document Field Extraction System
Extracts 6 fields from tractor invoice/quotation documents.

Usage:
    python executable.py <input_pdf_or_image> [--output result.json]

Input: PDF or Image (PNG/JPG)
Output: JSON with extracted fields
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from ultralytics import YOLO

# Paths to models (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
VLM_MODEL_PATH = SCRIPT_DIR / "models" / "qwen2.5-vl-7b-instruct"
VLM_LORA_PATH = SCRIPT_DIR / "vlm_lora_out"
YOLO_MODEL_PATH = SCRIPT_DIR / "runs" / "detect" / "stamp_sig_final" / "weights" / "best.pt"


def pdf_to_image(pdf_path: Path) -> Image.Image:
    """Convert PDF to image using pdf2image."""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=150)
        return images[0].convert("RGB")
    except ImportError:
        print("WARNING: pdf2image not installed. Trying alternative method...")
        # Fallback using PyMuPDF
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except ImportError:
            raise ImportError("Please install pdf2image or PyMuPDF: pip install pdf2image pymupdf")


def load_image(input_path: Path) -> Image.Image:
    """Load image from file or convert PDF to image."""
    suffix = input_path.suffix.lower()

    if suffix == ".pdf":
        return pdf_to_image(input_path)
    elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        return Image.open(input_path).convert("RGB")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from VLM response text."""
    # Try 1: Look for quoted JSON string first: "{ ... }"
    quoted_match = re.search(r'"(\{.*?\})"', text, re.DOTALL)
    if quoted_match:
        try:
            escaped_json = quoted_match.group(1)
            unescaped = escaped_json.replace('\\"', '"').replace('\\\\', '\\')
            return json.loads(unescaped)
        except Exception:
            pass

    # Try 2: Look for unquoted {...} block
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class DocumentExtractor:
    """Main extraction pipeline combining VLM and YOLO."""

    def __init__(self, device: str = "0", load_4bit: bool = True):
        self.device = f"cuda:{device}" if device != "cpu" else "cpu"
        self.load_4bit = load_4bit
        self.vlm_model = None
        self.vlm_processor = None
        self.yolo_model = None

    def load_models(self):
        """Load VLM and YOLO models."""
        print("Loading models...")
        start = time.time()

        # Load VLM
        print(f"  Loading VLM from {VLM_MODEL_PATH}...")
        self.vlm_processor = AutoProcessor.from_pretrained(
            str(VLM_MODEL_PATH), trust_remote_code=True, use_fast=False
        )

        quant_cfg = None
        device_map = self.device
        if self.load_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            device_map = "auto"

        self.vlm_model = AutoModelForImageTextToText.from_pretrained(
            str(VLM_MODEL_PATH),
            torch_dtype=torch.float16,
            device_map=device_map,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )

        # Load LoRA adapter if available
        if VLM_LORA_PATH.exists():
            print(f"  Loading LoRA adapter from {VLM_LORA_PATH}...")
            self.vlm_model = PeftModel.from_pretrained(self.vlm_model, str(VLM_LORA_PATH))

        # Load YOLO
        print(f"  Loading YOLO from {YOLO_MODEL_PATH}...")
        self.yolo_model = YOLO(str(YOLO_MODEL_PATH))

        print(f"  Models loaded in {time.time() - start:.1f}s")

    def extract_text_fields(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text fields using VLM."""
        system_prompt = (
            "You are a document extraction model. Read the tractor invoice/quotation image and return JSON with:\n"
            "dealer_name (string), model_name (string), horse_power (ONLY the numeric value, e.g. 45 not '45 HP'), "
            "asset_cost (ONLY the numeric value, no commas or currency symbols). "
            "If a field is missing, use empty string or null. Do not include units or text in numeric fields."
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Return JSON only."}]},
        ]

        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.vlm_processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.vlm_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.vlm_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.vlm_processor.tokenizer.pad_token_id,
            )

        response = self.vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        parsed = extract_json(response) or {}

        return {
            "dealer_name": parsed.get("dealer_name", ""),
            "model_name": parsed.get("model_name", ""),
            "horse_power": parsed.get("horse_power"),
            "asset_cost": parsed.get("asset_cost"),
        }

    def extract_bboxes(self, image: Image.Image) -> Dict[str, Any]:
        """Extract signature and stamp bounding boxes using YOLO."""
        # Save temp image for YOLO
        temp_path = Path("/tmp/temp_extract.png")
        image.save(temp_path)

        results = self.yolo_model.predict(str(temp_path), conf=0.25, verbose=False)

        sig_box, sig_conf = None, 0.0
        stamp_box, stamp_conf = None, 0.0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                if cls == 0 and conf > sig_conf:  # signature
                    sig_box, sig_conf = xyxy, conf
                elif cls == 1 and conf > stamp_conf:  # stamp
                    stamp_box, stamp_conf = xyxy, conf

        return {
            "signature": {
                "present": sig_box is not None,
                "bbox": sig_box,
                "conf": sig_conf
            },
            "stamp": {
                "present": stamp_box is not None,
                "bbox": stamp_box,
                "conf": stamp_conf
            }
        }

    def extract(self, input_path: Path) -> Dict[str, Any]:
        """Full extraction pipeline."""
        start_time = time.time()

        # Load image
        image = load_image(input_path)

        # Extract fields
        text_fields = self.extract_text_fields(image)
        bbox_fields = self.extract_bboxes(image)

        processing_time = time.time() - start_time

        # Compute confidence (average of available confidences)
        confs = []
        if bbox_fields["signature"]["conf"]:
            confs.append(bbox_fields["signature"]["conf"])
        if bbox_fields["stamp"]["conf"]:
            confs.append(bbox_fields["stamp"]["conf"])
        avg_conf = sum(confs) / len(confs) if confs else 0.5

        return {
            "doc_id": input_path.stem,
            "fields": {
                "dealer_name": text_fields["dealer_name"],
                "model_name": text_fields["model_name"],
                "horse_power": text_fields["horse_power"],
                "asset_cost": text_fields["asset_cost"],
                "signature": bbox_fields["signature"],
                "stamp": bbox_fields["stamp"],
            },
            "confidence": round(avg_conf, 4),
            "processing_time_sec": round(processing_time, 2),
            "cost_estimate_usd": 0.002  # Estimated cost per document
        }


def main():
    parser = argparse.ArgumentParser(description="Extract fields from invoice documents")
    parser.add_argument("input", help="Input PDF or image file")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file (default: stdout)")
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize extractor
    extractor = DocumentExtractor(device=args.device, load_4bit=not args.no_4bit)
    extractor.load_models()

    # Extract fields
    result = extractor.extract(input_path)

    # Output
    output_json = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Results saved to: {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
