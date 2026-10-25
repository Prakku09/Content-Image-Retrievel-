"""Dataset-agnostic image metadata and duplicate checks for external CBIR runs."""

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    relative_path: str
    label: str
    split: str
    identity: str


def load_metadata(dataset_root, metadata_file):
    """Load a CSV with image_path,label,split and optional identity columns."""
    root = Path(dataset_root).resolve()
    metadata_path = Path(metadata_file)
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    metadata_path = metadata_path.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    records = []
    seen_paths = set()
    identity_splits = {}
    identity_labels = {}
    with metadata_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"image_path", "label", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Metadata must contain image_path, label, and split columns."
            )
        for line_number, row in enumerate(reader, start=2):
            relative_path = (row.get("image_path") or "").strip().replace("\\", "/")
            label = (row.get("label") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not relative_path or not label:
                raise ValueError(f"Missing image path or label at CSV line {line_number}.")
            if split not in {"gallery", "query"}:
                raise ValueError(
                    f"Split must be gallery or query at CSV line {line_number}."
                )
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Image path escapes dataset root: {relative_path}")
            image_path = (root / relative).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError(f"Image path escapes dataset root: {relative_path}")
            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                raise ValueError(f"Unsupported image extension: {relative_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in metadata is missing: {image_path}")
            normalized_relative = image_path.relative_to(root).as_posix()
            if normalized_relative in seen_paths:
                raise ValueError(f"Duplicate image path in metadata: {normalized_relative}")
            seen_paths.add(normalized_relative)
            identity = (row.get("identity") or normalized_relative).strip()
            if not identity:
                raise ValueError(f"Empty identity at CSV line {line_number}.")
            previous_split = identity_splits.setdefault(identity, split)
            if previous_split != split:
                raise ValueError(
                    f"Identity {identity!r} occurs in both gallery and query splits."
                )
            previous_label = identity_labels.setdefault(identity, label)
            if previous_label != label:
                raise ValueError(
                    f"Identity {identity!r} maps to multiple labels."
                )
            records.append(ImageRecord(
                image_path=image_path,
                relative_path=normalized_relative,
                label=label,
                split=split,
                identity=identity,
            ))

    if not records:
        raise ValueError("Metadata contains no image records.")
    if not any(record.split == "gallery" for record in records):
        raise ValueError("Metadata contains no gallery images.")
    if not any(record.split == "query" for record in records):
        raise ValueError("Metadata contains no query images.")
    if not {record.label for record in records if record.split == "gallery"}.issuperset(
        {record.label for record in records if record.split == "query"}
    ):
        raise ValueError("Every query label must be represented in the gallery.")
    return records


def read_oxford_pet_splits(dataset_root):
    """Read official Oxford-IIIT Pet trainval/test lists as gallery/query records."""
    root = Path(dataset_root).resolve()
    image_root = root / "images"
    annotation_root = root / "annotations"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Oxford Pet images directory not found: {image_root}")

    records = []
    class_names = {}
    class_breeds = {}
    breed_names_by_id = {}
    identity_splits = {}
    for annotation_name, split in (
        ("trainval.txt", "gallery"),
        ("test.txt", "query"),
    ):
        annotation_path = annotation_root / annotation_name
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Oxford Pet split file not found: {annotation_path}")
        with annotation_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                values = line.split()
                if not values:
                    continue
                if len(values) != 4:
                    raise ValueError(
                        f"Expected 4 fields in {annotation_name}:{line_number}."
                    )
                stem, class_id, species_id, breed_id = values
                if not all(value.isdigit() for value in (class_id, species_id, breed_id)):
                    raise ValueError(
                        f"Invalid numeric identifier in {annotation_name}:{line_number}."
                    )
                breed_name = stem.rsplit("_", 1)[0].lower()
                if int(class_id) not in range(1, 38):
                    raise ValueError(
                        f"Invalid class ID in {annotation_name}:{line_number}."
                    )
                if int(species_id) not in (1, 2):
                    raise ValueError(
                        f"Invalid species ID in {annotation_name}:{line_number}."
                    )
                if int(breed_id) not in (range(1, 13) if species_id == "1" else range(1, 26)):
                    raise ValueError(
                        f"Invalid breed ID in {annotation_name}:{line_number}."
                    )
                existing_name = class_names.setdefault(class_id, breed_name)
                if existing_name != breed_name:
                    raise ValueError(
                        f"Class ID {class_id} maps to multiple breed names."
                    )
                existing_breed = class_breeds.setdefault(class_id, breed_id)
                if existing_breed != breed_id:
                    raise ValueError(
                        f"Class ID {class_id} maps to multiple breed IDs."
                    )
                breed_key = (species_id, breed_id)
                existing_breed_name = breed_names_by_id.setdefault(
                    breed_key, breed_name
                )
                if existing_breed_name != breed_name:
                    raise ValueError(
                        f"Species/breed ID {breed_key} maps to multiple breed names."
                    )
                previous_split = identity_splits.setdefault(stem, split)
                if previous_split != split:
                    raise ValueError(
                        f"Oxford image identity {stem!r} appears in both official splits."
                    )
                relative_path = Path("images") / f"{stem}.jpg"
                image_path = root / relative_path
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Oxford split image is missing: {image_path}"
                    )
                records.append(ImageRecord(
                    image_path=image_path.resolve(),
                    relative_path=relative_path.as_posix(),
                    label=breed_name,
                    split=split,
                    identity=stem,
                ))

    if len(class_names) != 37:
        raise ValueError(
            f"Expected the 37 Oxford Pet breeds, found {len(class_names)}."
        )
    if {int(class_id) for class_id in class_names} != set(range(1, 38)):
        raise ValueError("Oxford Pet class IDs must cover the range 1 through 37.")
    if len({record.identity for record in records}) != len(records):
        raise ValueError("Oxford split files contain repeated image identities.")
    if not {r.label for r in records if r.split == "gallery"}.issuperset(
        {r.label for r in records if r.split == "query"}
    ):
        raise ValueError("Every Oxford query breed must be represented in the gallery.")
    return records


def _perceptual_hash(path):
    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    coefficients = dctn(pixels, type=2, norm="ortho")[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def detect_duplicates(records, near_distance=4):
    """Detect exact duplicates globally and perceptual candidates across splits."""
    if near_distance < 0:
        raise ValueError("near_distance must be non-negative.")
    content_hashes = {}
    exact_groups = defaultdict(list)
    perceptual_hashes = {}
    for record in records:
        digest = hashlib.sha256(record.image_path.read_bytes()).hexdigest()
        content_hashes[record.relative_path] = digest
        exact_groups[digest].append(record)
        perceptual_hashes[record.relative_path] = _perceptual_hash(
            record.image_path
        )

    exact_pairs = []
    exact_cross_split_pairs = []
    for group in exact_groups.values():
        if len(group) > 1:
            for left, right in combinations(group, 2):
                pair = {
                    "left_path": left.relative_path,
                    "right_path": right.relative_path,
                    "sha256": content_hashes[left.relative_path],
                    "cross_split": left.split != right.split,
                }
                exact_pairs.append(pair)
                if pair["cross_split"]:
                    exact_cross_split_pairs.append(pair)

    gallery = [r for r in records if r.split == "gallery"]
    queries = [r for r in records if r.split == "query"]
    near_pairs = []
    for query in queries:
        query_hash = perceptual_hashes[query.relative_path]
        for gallery_record in gallery:
            if content_hashes[query.relative_path] == content_hashes[
                gallery_record.relative_path
            ]:
                continue
            distance = (
                query_hash ^ perceptual_hashes[gallery_record.relative_path]
            ).bit_count()
            if distance <= near_distance:
                near_pairs.append({
                    "gallery_path": gallery_record.relative_path,
                    "query_path": query.relative_path,
                    "hamming_distance": distance,
                    "same_label": gallery_record.label == query.label,
                })

    return {
        "content_hashes": content_hashes,
        "perceptual_hashes": perceptual_hashes,
        "exact_duplicate_pairs": exact_pairs,
        "exact_cross_split_pairs": exact_cross_split_pairs,
        "near_duplicate_candidates_cross_split": near_pairs,
        "near_hash_threshold": near_distance,
    }


def dataset_fingerprint(records, content_hashes, source_files=()):
    digest = hashlib.sha256()
    for source_file in sorted(Path(p).resolve() for p in source_files):
        digest.update(source_file.name.encode("utf-8"))
        digest.update(source_file.read_bytes())
    for record in sorted(records, key=lambda item: item.relative_path):
        digest.update(record.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.split.encode("ascii"))
        digest.update(b"\0")
        digest.update(content_hashes[record.relative_path].encode("ascii"))
    return digest.hexdigest()
