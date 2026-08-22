import sys
from pathlib import Path
import config

def perform_validation():
    path = config.DATASET_PATH
    exists = path.exists() and path.is_dir()
    
    if not exists:
        print(f"Dataset path: {path}")
        print(f"Dataset exists: {exists}")
        print("Number of categories: 0")
        print("Category names: None")
        print("Images per category: None")
        print("Total images: 0")
        print(f"Image extensions: {', '.join(sorted(config.IMAGE_EXTENSIONS))}")
        print("Corel-1K validation: NOT READY")
        print("\n[ERROR] Dataset path does not exist or is not a directory.")
        sys.exit(1)
        
    # Scan directory recursively for all supported image files
    image_paths = []
    for ext in config.IMAGE_EXTENSIONS:
        image_paths.extend(path.rglob(f"*{ext}"))
        image_paths.extend(path.rglob(f"*{ext.upper()}"))
        
    # Resolve absolute paths and deduplicate to avoid duplicates on case-insensitive OS
    image_paths = list(set(p.resolve() for p in image_paths))
    
    # Identify category folders (using the immediate parent folder name of each image)
    category_to_images = {}
    for p in image_paths:
        cat_name = p.parent.name
        if cat_name not in category_to_images:
            category_to_images[cat_name] = []
        category_to_images[cat_name].append(p)
        
    # Sort detected category names
    detected_cats = sorted(category_to_images.keys())
    total_images = len(image_paths)
    
    # Format images per category string
    images_per_cat_str = ", ".join([f"{cat}: {len(category_to_images[cat])}" for cat in detected_cats])
    
    # Validate Corel-1K expectations
    validation_passed = True
    errors = []
    
    # 1. Expect exactly 10 categories
    if len(detected_cats) != 10:
        validation_passed = False
        errors.append(f"Expected exactly 10 category directories, but detected {len(detected_cats)}: {detected_cats}")
        
    # 2. Check if detected categories match expected categories
    expected_set = set(config.EXPECTED_CATEGORIES)
    detected_set = set(detected_cats)
    
    missing_cats = expected_set - detected_set
    extra_cats = detected_set - expected_set
    
    if missing_cats:
        validation_passed = False
        errors.append(f"Missing expected categories: {missing_cats}")
    if extra_cats:
        validation_passed = False
        errors.append(f"Unexpected category folders detected: {extra_cats}")
        
    # 3. Check if total number of images is approximately 1000
    if not (950 <= total_images <= 1050):
        validation_passed = False
        errors.append(f"Expected approximately 1000 images, but detected {total_images} images.")
        
    # Print the exact requested final terminal output
    print(f"Dataset path: {path}")
    print(f"Dataset exists: {exists}")
    print(f"Number of categories: {len(detected_cats)}")
    print(f"Category names: {', '.join(detected_cats) if detected_cats else 'None'}")
    print(f"Images per category: {images_per_cat_str if images_per_cat_str else 'None'}")
    print(f"Total images: {total_images}")
    print(f"Image extensions: {', '.join(sorted(config.IMAGE_EXTENSIONS))}")
    
    if validation_passed:
        print("Corel-1K validation: READY FOR PART 1")
        sys.exit(0)
    else:
        print("Corel-1K validation: NOT READY")
        print("\n=== DETECTED ANOMALIES ===")
        for err in errors:
            print(f"- {err}")
        print("\n=== DETECTED DIRECTORY STRUCTURE ===")
        for item in sorted(path.iterdir()):
            if item.is_dir():
                sub_files = len([f for f in item.glob("**/*") if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS])
                print(f"Folder '{item.name}' contains {sub_files} images.")
            else:
                print(f"File '{item.name}'")
        sys.exit(1)

if __name__ == "__main__":
    perform_validation()
