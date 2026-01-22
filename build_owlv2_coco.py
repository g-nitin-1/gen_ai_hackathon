"""
Build a COCO-style dataset for signature/stamp from current gold labels.
Reads annotations/seed/done/{images,labels}, writes owlv2_data/annotations.json and copies images.
"""

import json
from pathlib import Path

def main():
    img_dir = Path("annotations/seed/done/images")
    lab_dir = Path("annotations/seed/done/labels")
    out_ann = Path("owlv2_data/annotations.json")
    out_img = Path("owlv2_data/images")
    out_img.mkdir(parents=True, exist_ok=True)
    cats=[{"id":1,"name":"signature"},{"id":2,"name":"stamp"}]
    imgs=[]; anns=[]; img_id=1; ann_id=1
    for lab in lab_dir.glob("*.json"):
        img_path = img_dir / f"{lab.stem}.png"
        if not img_path.exists():
            continue
        # copy image into owlv2_data/images
        dst_path = out_img / img_path.name
        if not dst_path.exists():
            try:
                import shutil
                shutil.copy2(img_path, dst_path)
            except Exception as e:
                print(f"failed to copy {img_path}: {e}")
                continue
        obj=json.loads(lab.read_text()); fields=obj.get("fields",{})
        try:
            from PIL import Image
            W,H = Image.open(dst_path).size
        except Exception:
            W=H=None
        imgs.append({"id":img_id,"file_name":dst_path.name,"width":W,"height":H})
        for name,cid in [("signature",1),("stamp",2)]:
            bb=fields.get(name,{}).get("bbox")
            if bb:
                x1,y1,x2,y2=bb
                anns.append({"id":ann_id,"image_id":img_id,"category_id":cid,
                             "bbox":[x1,y1,x2-x1,y2-y1],
                             "area":(x2-x1)*(y2-y1),"iscrowd":0})
                ann_id+=1
        img_id+=1
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    out_data = {"images":imgs,"annotations":anns,"categories":cats}
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    out_ann.write_text(json.dumps(out_data, indent=2))
    print(f"wrote {len(imgs)} images, {len(anns)} anns to {out_ann}")
    if len(imgs)==0 or len(anns)==0:
        print("WARNING: no images or no annotations found. Check done labels for signature/stamp bboxes and matching PNGs.")

if __name__ == "__main__":
    main()
