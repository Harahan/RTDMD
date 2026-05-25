"""
GenEval compositional evaluation scorer.

Uses Mask2Former object detection + CLIP color classification to evaluate
compositional generation accuracy (object count, color, position).


NOTE: This scorer has heavy dependencies (mmdet, open_clip, clip_benchmark).
It is only loaded when dataset=="geneval" is configured.
"""

import json
import os
import sys
import time
import warnings
import logging
import urllib.request
import fcntl
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def _preserve_root_logger_state():
    """Keep RTDMD root logger settings stable across third-party side-effects."""
    root = logging.getLogger()
    level = root.level
    disabled = root.disabled
    propagate = root.propagate
    handlers = list(root.handlers)
    handler_levels = [(handler, handler.level) for handler in handlers]
    try:
        yield
    finally:
        root.setLevel(level)
        root.disabled = disabled
        root.propagate = propagate
        root.handlers[:] = handlers
        for handler, handler_level in handler_levels:
            handler.setLevel(handler_level)


warnings.filterwarnings("ignore")

import numpy as np
from collections import defaultdict
from PIL import Image, ImageOps
import torch
with _preserve_root_logger_state():
    import mmdet
    from mmdet.apis import inference_detector, init_detector
    import open_clip
    from clip_benchmark.metrics import zeroshot_classification as zsc

zsc.tqdm = lambda it, *args, **kwargs: it


def _resolve_geneval_mmdet_config_path() -> str:
    """Resolve Mask2Former config path across mmdet packaging layouts.

    Supports:
    - explicit override: RTDMD_GENEVAL_MMDET_CONFIG_PATH
    - pip mmdet package: mmdet/.mim/configs/...
    - source tree layouts with top-level configs/...
    """
    env_path = os.environ.get("RTDMD_GENEVAL_MMDET_CONFIG_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(
            f"RTDMD_GENEVAL_MMDET_CONFIG_PATH points to missing file: {candidate}"
        )

    mmdet_root = Path(mmdet.__file__).resolve().parent
    site_packages_root = mmdet_root.parent
    config_filenames = [
        # legacy naming used by old DiffusionNFT/GenEval setup
        "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
        # current mmdet package naming
        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py",
    ]

    candidate_dirs = [
        mmdet_root / ".mim" / "configs" / "mask2former",
        site_packages_root / "mmdet" / ".mim" / "configs" / "mask2former",
        site_packages_root / "configs" / "mask2former",
    ]

    for config_dir in candidate_dirs:
        for filename in config_filenames:
            candidate = config_dir / filename
            if candidate.is_file():
                return str(candidate)

    raise FileNotFoundError(
        "Cannot find Mask2Former config for GenEval. Checked:\n"
        + "\n".join(str(d / f) for d in candidate_dirs for f in config_filenames)
    )


def _resolve_geneval_mmdet_checkpoint_path(config_path: str) -> str:
    """Resolve/download checkpoint that matches selected Mask2Former config."""
    from rtdmd.rewards.reward_ckpt_path import CKPT_PATH

    env_path = os.environ.get("RTDMD_GENEVAL_MMDET_CKPT_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(
            f"RTDMD_GENEVAL_MMDET_CKPT_PATH points to missing file: {candidate}"
        )

    ckpt_root = Path(CKPT_PATH).expanduser()
    ckpt_root.mkdir(parents=True, exist_ok=True)
    cfg_name = Path(config_path).name

    # mmdet v3 config/checkpoint pair (matches config filename with 8xb2 naming).
    modern_filename = (
        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
    )
    modern_url = (
        "https://download.openmmlab.com/mmdetection/v3.0/mask2former/"
        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/"
        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
    )

    # legacy v2 filename used in older setups
    legacy_filename = (
        "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
    )
    legacy_path = ckpt_root / legacy_filename

    # Project default: prefer the legacy checkpoint if it already exists locally.
    # This avoids unnecessary redownloads and matches the commonly provisioned
    # reward_ckpts layout in this workspace.
    if legacy_path.is_file():
        if "8xb2-lsj-50e_coco" in cfg_name:
            warnings.warn(
                f"Using legacy Mask2Former checkpoint as default with modern config: {legacy_path}. "
                "Set RTDMD_GENEVAL_MMDET_CKPT_PATH to override.",
                stacklevel=2,
            )
        return str(legacy_path)

    if "8xb2-lsj-50e_coco" in cfg_name:
        ckpt_path = ckpt_root / modern_filename
        if not ckpt_path.is_file():
            _download_file_once(modern_url, ckpt_path)
        return str(ckpt_path)

    # Legacy config path keeps legacy filename to avoid key-space mismatch.
    if "lsj_8x2_50e_coco" in cfg_name:
        modern_path = ckpt_root / modern_filename
        if modern_path.is_file():
            warnings.warn(
                f"Using modern checkpoint for legacy config: {modern_path}",
                stacklevel=2,
            )
            return str(modern_path)
        _download_file_once(modern_url, modern_path)
        warnings.warn(
            f"Legacy checkpoint not found. Downloaded modern checkpoint instead: {modern_path}",
            stacklevel=2,
        )
        return str(modern_path)

    # Unknown config naming: prefer modern checkpoint.
    fallback_path = ckpt_root / modern_filename
    if not fallback_path.is_file():
        _download_file_once(modern_url, fallback_path)
    return str(fallback_path)


def _download_file_once(url: str, dst_path: Path) -> None:
    dst_path = dst_path.expanduser()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    lock_path = dst_path.with_suffix(dst_path.suffix + ".lock")
    with _file_lock(str(lock_path)):
        if dst_path.is_file():
            return
        print(f"download {url} to {dst_path}", file=sys.stderr)
        urllib.request.urlretrieve(url, str(tmp_path))
        os.replace(tmp_path, dst_path)


def _load_geneval_clip_model(device):
    """Load CLIP backbone for GenEval color classification with robust fallbacks."""
    env_arch = os.environ.get("RTDMD_GENEVAL_CLIP_ARCH", "").strip()
    env_pretrained = os.environ.get("RTDMD_GENEVAL_CLIP_PRETRAINED", "").strip()

    candidates: list[tuple[str, str]] = []
    if env_arch:
        candidates.append((env_arch, env_pretrained or "openai"))
    else:
        # Default path first, then compatibility fallbacks.
        candidates.extend(
            [
                ("ViT-L-14", "openai"),
                ("ViT-L-14-quickgelu", "openai"),
                ("ViT-L-14", "laion2b_s32b_b82k"),
            ]
        )

    errors: list[str] = []
    for clip_arch, pretrained_tag in candidates:
        try:
            clip_model, _, transform = open_clip.create_model_and_transforms(
                clip_arch,
                pretrained=pretrained_tag,
                device=device,
            )
            tokenizer = open_clip.get_tokenizer(clip_arch)
            if (clip_arch, pretrained_tag) != candidates[0]:
                warnings.warn(
                    f"GenEval CLIP fallback activated: arch={clip_arch}, pretrained={pretrained_tag}",
                    stacklevel=2,
                )
            return clip_model, transform, tokenizer
        except Exception as e:
            errors.append(f"{clip_arch}/{pretrained_tag}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Failed to initialize GenEval CLIP backbone. Tried "
        + ", ".join(f"{a}/{p}" for a, p in candidates)
        + ". Errors: "
        + " | ".join(errors)
    )


def _resolve_geneval_assets_dir() -> str:
    """Resolve GenEval assets directory (object_names.txt).

    Defaults to a sibling ``assets/`` folder next to this module. Override
    by setting the ``RTDMD_GENEVAL_ASSETS_DIR`` environment variable to point
    at a directory containing ``object_names.txt``.
    """
    env_path = os.environ.get("RTDMD_GENEVAL_ASSETS_DIR", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parent / "assets")

    for p in candidates:
        if (p / "object_names.txt").is_file():
            return str(p)

    raise FileNotFoundError(
        "Cannot locate GenEval assets/object_names.txt. Checked:\n"
        + "\n".join(str(p / "object_names.txt") for p in candidates)
        + "\nSet RTDMD_GENEVAL_ASSETS_DIR to override."
    )

@contextmanager
def _file_lock(lock_path: str):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_geneval(DEVICE):
    def timed(fn):
        def wrapper(*args, **kwargs):
            startt = time.time()
            result = fn(*args, **kwargs)
            endt = time.time()
            print(
                f"Function {fn.__name__!r} executed in {endt - startt:.3f}s",
                file=sys.stderr,
            )
            return result

        return wrapper

    # Load models

    @timed
    def load_models():
        CONFIG_PATH = _resolve_geneval_mmdet_config_path()
        _CKPT_PATH = _resolve_geneval_mmdet_checkpoint_path(CONFIG_PATH)
        object_detector = init_detector(CONFIG_PATH, _CKPT_PATH, device=DEVICE)

        clip_model, transform, tokenizer = _load_geneval_clip_model(DEVICE)

        # Load object names from assets directory (same location as in DiffusionNFT)
        assets_dir = _resolve_geneval_assets_dir()
        with open(os.path.join(assets_dir, "object_names.txt")) as cls_file:
            classnames = [line.strip() for line in cls_file]

        return object_detector, (clip_model, transform, tokenizer), classnames

    # NOTE:
    # color_classification runs inside reward scoring, which may itself be invoked
    # from async worker threads (e.g., GRPO ThreadPoolExecutor). Creating
    # DataLoader worker processes in that context is fragile and can trigger
    # "DataLoader worker exited unexpectedly" at unrelated call sites due
    # global signal handling. Keep workers off by default for stability.
    _env_workers = os.environ.get("RTDMD_GENEVAL_COLOR_WORKERS", "0")
    try:
        color_workers = max(0, int(_env_workers))
    except ValueError:
        color_workers = 0

    COLORS = [
        "red", "orange", "yellow", "green", "blue",
        "purple", "pink", "brown", "black", "white",
    ]
    COLOR_CLASSIFIERS = {}

    # Evaluation parts

    class ImageCrops(torch.utils.data.Dataset):
        def __init__(self, image: Image.Image, objects):
            self._image = image.convert("RGB")
            bgcolor = "#999"
            self._blank = Image.new("RGB", image.size, color=bgcolor)
            self._objects = objects

        def __len__(self):
            return len(self._objects)

        def __getitem__(self, index):
            box, mask = self._objects[index]
            if mask is not None:
                assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                    index, self._image.size[::-1], mask.shape,
                )
                image = Image.composite(self._image, self._blank, Image.fromarray(mask))
            else:
                image = self._image
            image = image.crop(box[:4])
            return (transform(image), 0)

    def color_classification(image, bboxes, classname):
        if classname not in COLOR_CLASSIFIERS:
            COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                clip_model, tokenizer, COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                str(DEVICE),
            )
        clf = COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes),
            batch_size=16,
            num_workers=color_workers,
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(clip_model, clf, dataloader, str(DEVICE))
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def compute_iou(box_a, box_b):
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        i_area = area_fn([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3]),
        ])
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    def relative_position(obj_a, obj_b):
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def evaluate(image, objects, metadata):
        correct = True
        reason = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[:req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                if "color" in req:
                    colors = color_classification(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)

    def evaluate_reward(image, objects, metadata):
        correct = True
        reason = []
        rewards = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])
            if len(found_objects) != req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}=={req['count']}, found {len(found_objects)}")
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = color_classification(image, found_objects, classname)
                    rewards.append(1 - abs(req["count"] - colors.count(req["color"])) / req["count"])
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        rewards.append(1.0)
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, "\n".join(reason)

    def evaluate_image(image_pils, metadatas, only_strict):
        results = inference_detector(object_detector, [np.array(image_pil) for image_pil in image_pils])
        ret = []
        for result, image_pil, metadata in zip(results, image_pils, metadatas):
            bbox = result[0] if isinstance(result, tuple) else result
            segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            image = ImageOps.exif_transpose(image_pil)
            detected = {}
            confidence_threshold = THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
            for index, classname in enumerate(classnames):
                ordering = np.argsort(bbox[index][:, 4])[::-1]
                ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
                ordering = ordering[:MAX_OBJECTS].tolist()
                detected[classname] = []
                while ordering:
                    max_obj = ordering.pop(0)
                    detected[classname].append((
                        bbox[index][max_obj],
                        None if segm is None else segm[index][max_obj],
                    ))
                    ordering = [
                        obj for obj in ordering
                        if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
                    ]
                if not detected[classname]:
                    del detected[classname]
            is_strict_correct, score, reason = evaluate_reward(image, detected, metadata)
            if only_strict:
                is_correct = False
            else:
                is_correct, _ = evaluate(image, detected, metadata)
            ret.append({
                "tag": metadata["tag"],
                "prompt": metadata["prompt"],
                "correct": is_correct,
                "strict_correct": is_strict_correct,
                "score": score,
                "reason": reason,
                "metadata": json.dumps(metadata),
                "details": json.dumps({key: [box.tolist() for box, _ in value] for key, value in detected.items()}),
            })
        return ret

    with _preserve_root_logger_state():
        object_detector, (clip_model, transform, tokenizer), classnames = load_models()
    THRESHOLD = 0.3
    COUNTING_THRESHOLD = 0.9
    MAX_OBJECTS = 16
    NMS_THRESHOLD = 1.0
    POSITION_THRESHOLD = 0.1

    @torch.no_grad()
    def compute_geneval(images, metadatas, only_strict=False):
        required_keys = [
            "single_object", "two_object", "counting",
            "colors", "position", "color_attr",
        ]
        scores = []
        strict_rewards = []
        grouped_strict_rewards = defaultdict(list)
        rewards = []
        grouped_rewards = defaultdict(list)
        results = evaluate_image(images, metadatas, only_strict=only_strict)
        for result in results:
            strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
            scores.append(result["score"])
            rewards.append(1.0 if result["correct"] else 0.0)
            tag = result["tag"]
            for key in required_keys:
                if key != tag:
                    grouped_strict_rewards[key].append(-10.0)
                    grouped_rewards[key].append(-10.0)
                else:
                    grouped_strict_rewards[tag].append(1.0 if result["strict_correct"] else 0.0)
                    grouped_rewards[tag].append(1.0 if result["correct"] else 0.0)
        return (
            scores, rewards, strict_rewards,
            dict(grouped_rewards), dict(grouped_strict_rewards),
        )

    return compute_geneval
