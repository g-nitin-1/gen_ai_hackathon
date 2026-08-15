"""
IDFC GenAI Document Field Extraction - Gradio Web App
Upload a PDF or image to extract fields.

Usage:
    python app.py
    # Opens at http://localhost:7860
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from ultralytics import YOLO

# Paths to models
SCRIPT_DIR = Path(__file__).resolve().parent
VLM_MODEL_PATH = SCRIPT_DIR / "models" / "qwen2.5-vl-7b-instruct"
VLM_LORA_PATH = SCRIPT_DIR / "vlm_lora_final"
YOLO_MODEL_PATH = SCRIPT_DIR / "runs" / "detect" / "stamp_sig_final" / "weights" / "best.pt"

# Fallback paths (if final models not trained yet)
if not VLM_LORA_PATH.exists():
    VLM_LORA_PATH = SCRIPT_DIR / "vlm_lora_out"
if not YOLO_MODEL_PATH.exists():
    YOLO_MODEL_PATH = SCRIPT_DIR / "runs" / "detect" / "stamp_sig_v62" / "weights" / "best.pt"

# Global model holders
vlm_model = None
vlm_processor = None
yolo_model = None
models_loaded = False


def pdf_to_image(pdf_path: Path) -> Image.Image:
    """Convert PDF to image."""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=150)
        return images[0].convert("RGB")
    except ImportError:
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except ImportError:
            raise ImportError("Please install pdf2image or PyMuPDF")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from VLM response."""
    quoted_match = re.search(r'"(\{.*?\})"', text, re.DOTALL)
    if quoted_match:
        try:
            escaped_json = quoted_match.group(1)
            unescaped = escaped_json.replace('\\"', '"').replace('\\\\', '\\')
            return json.loads(unescaped)
        except:
            pass

    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return None


def load_models():
    """Load VLM and YOLO models."""
    global vlm_model, vlm_processor, yolo_model, models_loaded

    if models_loaded:
        return "Models already loaded!"

    yield "Loading VLM processor..."
    vlm_processor = AutoProcessor.from_pretrained(
        str(VLM_MODEL_PATH), trust_remote_code=True, use_fast=False
    )

    yield "Loading VLM model (4-bit quantization)..."
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    vlm_model = AutoModelForImageTextToText.from_pretrained(
        str(VLM_MODEL_PATH),
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )

    if VLM_LORA_PATH.exists():
        yield f"Loading LoRA adapter from {VLM_LORA_PATH.name}..."
        vlm_model = PeftModel.from_pretrained(vlm_model, str(VLM_LORA_PATH))

    yield "Loading YOLO model..."
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    models_loaded = True
    yield "All models loaded successfully!"


def draw_boxes(image: Image.Image, signature_box, stamp_box) -> Image.Image:
    """Draw bounding boxes on image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    if signature_box:
        x1, y1, x2, y2 = signature_box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1-20), "SIGNATURE", fill="green")

    if stamp_box:
        x1, y1, x2, y2 = stamp_box
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        draw.text((x1, y1-20), "STAMP", fill="blue")

    return img_draw


def extract_fields(file_path: str, progress=gr.Progress()):
    """Main extraction function."""
    global vlm_model, vlm_processor, yolo_model, models_loaded

    if not models_loaded:
        return None, "Please load models first!", None

    if file_path is None:
        return None, "Please upload a file!", None

    start_time = time.time()

    # Load image
    progress(0.1, desc="Loading image...")
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        image = pdf_to_image(file_path)
    else:
        image = Image.open(file_path).convert("RGB")

    # VLM extraction
    progress(0.3, desc="Extracting text fields with VLM...")
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

    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm_processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(vlm_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vlm_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=vlm_processor.tokenizer.pad_token_id,
        )

    response = vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    parsed = extract_json(response) or {}

    # YOLO detection
    progress(0.7, desc="Detecting signature and stamp...")
    temp_path = Path("/tmp/temp_gradio.png")
    image.save(temp_path)

    results = yolo_model.predict(str(temp_path), conf=0.25, verbose=False)

    sig_box, sig_conf = None, 0.0
    stamp_box, stamp_conf = None, 0.0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            if cls == 0 and conf > sig_conf:
                sig_box, sig_conf = xyxy, conf
            elif cls == 1 and conf > stamp_conf:
                stamp_box, stamp_conf = xyxy, conf

    progress(0.9, desc="Generating results...")

    # Draw boxes on image
    annotated_image = draw_boxes(image, sig_box, stamp_box)

    processing_time = time.time() - start_time

    # Build result
    result = {
        "doc_id": file_path.stem,
        "fields": {
            "dealer_name": parsed.get("dealer_name", ""),
            "model_name": parsed.get("model_name", ""),
            "horse_power": parsed.get("horse_power"),
            "asset_cost": parsed.get("asset_cost"),
            "signature": {
                "present": sig_box is not None,
                "bbox": sig_box,
                "conf": round(sig_conf, 4)
            },
            "stamp": {
                "present": stamp_box is not None,
                "bbox": stamp_box,
                "conf": round(stamp_conf, 4)
            }
        },
        "confidence": round((sig_conf + stamp_conf) / 2 if sig_conf or stamp_conf else 0.5, 4),
        "processing_time_sec": round(processing_time, 2),
        "cost_estimate_usd": 0.002
    }

    result_json = json.dumps(result, indent=2)

    # Create summary text
    summary = f"""
## Extraction Results

| Field | Value |
|-------|-------|
| **Dealer Name** | {parsed.get('dealer_name', 'N/A')} |
| **Model Name** | {parsed.get('model_name', 'N/A')} |
| **Horse Power** | {parsed.get('horse_power', 'N/A')} HP |
| **Asset Cost** | ₹{parsed.get('asset_cost', 'N/A'):,} |
| **Signature** | {'✅ Present' if sig_box else '❌ Not Found'} (conf: {sig_conf:.2f}) |
| **Stamp** | {'✅ Present' if stamp_box else '❌ Not Found'} (conf: {stamp_conf:.2f}) |

**Processing Time:** {processing_time:.2f} seconds
**Estimated Cost:** $0.002
"""

    return annotated_image, summary, result_json


# Create Gradio interface
with gr.Blocks(title="IDFC GenAI Document Extractor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚜 IDFC GenAI Document Field Extraction

    Upload a tractor invoice/quotation (PDF or Image) to extract:
    - Dealer Name
    - Model Name
    - Horse Power
    - Asset Cost
    - Signature (with bounding box)
    - Stamp (with bounding box)

    **Architecture:** Qwen2.5-VL-7B (LoRA fine-tuned) + YOLOv11
    """)

    with gr.Row():
        with gr.Column():
            load_btn = gr.Button("🔄 Load Models", variant="primary", size="lg")
            load_status = gr.Textbox(label="Status", interactive=False)

            file_input = gr.File(
                label="Upload Invoice (PDF or Image)",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath"
            )

            extract_btn = gr.Button("🔍 Extract Fields", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="Annotated Document", type="pil")

    with gr.Row():
        with gr.Column():
            summary_output = gr.Markdown(label="Summary")
        with gr.Column():
            json_output = gr.Code(label="JSON Output", language="json")

    # Event handlers
    load_btn.click(
        fn=load_models,
        outputs=load_status
    )

    extract_btn.click(
        fn=extract_fields,
        inputs=[file_input],
        outputs=[output_image, summary_output, json_output]
    )

    gr.Markdown("""
    ---
    **Team:** Convolve 4.0 | **Hackathon:** IDFC GenAI 2024
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
