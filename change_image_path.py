import json
from pathlib import Path

# Base folder
train_folder = Path("mini_data/train")

# Prefix to replace
old_prefix = "/mnt/localssd/code/data/yochameleon-data/"
new_prefix = "mini_data/"

json_files = list(train_folder.rglob("*.json"))
print(json_files)
# Loop over all subfolders in train/

for json_file in json_files:
    with open(json_file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping invalid JSON: {json_file}")
            continue

    # If the JSON contains a list of items
    if isinstance(data, list):
        for item in data:
            # Fix "image" key if exists (list of paths or chars)
            if "image" in item and isinstance(item["image"], list):
                new_images = []
                for img_path in item["image"]:
                    if isinstance(img_path, str) and img_path.startswith(old_prefix):
                        img_path = img_path.replace(old_prefix, new_prefix)
                    new_images.append(img_path)
                item["image"] = new_images

            # Fix "image_path" key if exists
            if "image_path" in item and isinstance(item["image_path"], str):
                if item["image_path"].startswith(old_prefix):
                    item["image_path"] = item["image_path"].replace(old_prefix, new_prefix)

    # Save back to the same file (overwrite)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Updated {json_file}")
