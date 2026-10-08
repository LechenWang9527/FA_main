from pathlib import Path
import json

DATA = Path("/root/autodl-tmp/DATA")
CLIP = Path("/root/autodl-tmp/clip_weights/ViT-B-16.pt")

paths = {
    "EuroSAT image dir": DATA / "eurosat" / "2750",
    "EuroSAT split": DATA / "eurosat" / "split_zhou_EuroSAT.json",
    "DTD image dir": DATA / "dtd" / "images",
    "DTD split": DATA / "dtd" / "split_zhou_DescribableTextures.json",
    "CLIP weight": CLIP,
}

print("==== Path Check ====")
all_ok = True
for name, p in paths.items():
    ok = p.exists()
    all_ok = all_ok and ok
    print(f"{name}: {p} -> {'OK' if ok else 'MISSING'}")

print("\n==== EuroSAT Classes ====")
eurosat_dir = DATA / "eurosat" / "2750"
if eurosat_dir.exists():
    classes = sorted([x.name for x in eurosat_dir.iterdir() if x.is_dir()])
    print(f"num_classes = {len(classes)}")
    print(classes)

print("\n==== DTD Classes ====")
dtd_dir = DATA / "dtd" / "images"
if dtd_dir.exists():
    classes = sorted([x.name for x in dtd_dir.iterdir() if x.is_dir()])
    print(f"num_classes = {len(classes)}")
    print(classes[:20], "..." if len(classes) > 20 else "")

print("\n==== Split JSON Check ====")
for split_name, split_path in [
    ("EuroSAT", DATA / "eurosat" / "split_zhou_EuroSAT.json"),
    ("DTD", DATA / "dtd" / "split_zhou_DescribableTextures.json"),
]:
    if split_path.exists():
        try:
            with open(split_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            print(f"{split_name} split keys:", obj.keys())
            for k in ["train", "val", "test"]:
                if k in obj:
                    print(f"  {k}: {len(obj[k])} samples")
        except Exception as e:
            print(f"{split_name} split read error:", e)

if not all_ok:
    raise SystemExit("\nSome paths are missing. Fix dataset folders first.")
else:
    print("\nAll basic dataset paths look good.")
