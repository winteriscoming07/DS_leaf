"""Inference wrapper: same preprocessing + frozen validation-derived threshold as the evaluated model.
Usage: python -m src.predict IMAGE [IMAGE ...] [--model mobilenet|hog_svm]"""
import argparse
import json

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.data import load_config
from src.open_set import UNKNOWN_LABEL, decide
from src.preprocessing import eval_transform, hog_features, load_rgb


class LeafPredictor:
    def __init__(self, model="mobilenet", results_path="artifacts/metrics/open_set_results.json", config="configs/experiment.json"):
        cfg = load_config(config)
        self.classes = list(cfg["known"])
        self.model_name = model
        self.threshold = json.load(open(results_path))[model]["threshold"]  # frozen at evaluation time
        if model == "mobilenet":
            from src.train_mobilenet import build_model, get_device
            self.device = get_device()
            ck = torch.load("artifacts/checkpoints/mobilenet_v3_small.pth", map_location=self.device)
            self.net = build_model(len(self.classes), pretrained=False).to(self.device)
            self.net.load_state_dict(ck["state_dict"]); self.net.eval()
            self.tf = eval_transform(ck["image_size"])
        elif model == "hog_svm":
            self.svm = joblib.load("artifacts/checkpoints/hog_svm.joblib")
            self.hog_size = cfg["image_size_hog"]
        else:
            raise ValueError(f"unknown model {model!r}")

    @torch.no_grad()
    def probs(self, images):
        """images: list of PIL RGB images -> (n, 5) probabilities."""
        if self.model_name == "mobilenet":
            x = torch.stack([self.tf(im) for im in images]).to(self.device)
            return torch.softmax(self.net(x), 1).cpu().numpy()
        return self.svm.predict_proba(np.stack([hog_features(im, self.hog_size) for im in images]))

    def predict_images(self, images):
        return [dict(predicted=p, confidence=c, outcome=o, accepted=o != UNKNOWN_LABEL, threshold=self.threshold)
                for p, c, o in decide(self.probs(images), self.threshold, self.classes)]

    def predict_paths(self, paths):
        return self.predict_images([load_rgb(p) for p in paths])


MAX_UPLOAD_PIXELS = 50_000_000


def open_upload(file_like):
    """Decode an uploaded file to an RGB PIL image (EXIF orientation applied, alpha/palette flattened).
    Raises ValueError with a user-readable message for non-images, empty or oversized files."""
    try:
        with Image.open(file_like) as im:
            if im.width * im.height > MAX_UPLOAD_PIXELS:
                raise ValueError("Image is too large (over 50 megapixels); please upload a smaller photo.")
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); im = Image.alpha_composite(bg, im)
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("This file could not be read as an image (JPG or PNG expected).") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="mobilenet", choices=["mobilenet", "hog_svm"])
    a = ap.parse_args()
    pr = LeafPredictor(a.model)
    failed = 0
    for path in a.images:
        try:
            with open(path, "rb") as f:
                img = open_upload(f)
        except (OSError, ValueError) as e:
            print(f"{path}: ERROR - {e}")
            failed += 1
            continue
        r = pr.predict_images([img])[0]
        print(f"{path}: {r['outcome']}  (best guess: {r['predicted']}, confidence {r['confidence']:.3f}, threshold {r['threshold']:.3f})")
    print("Note: rejection is a confidence-based safeguard, not proof that the plant is botanically unknown.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
