"""Model B: MobileNetV3-Small (ImageNet weights), frozen feature extractor, trained 5-way classifier head."""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.data import load_config, load_metadata
from src.preprocessing import eval_transform, load_rgb, train_transform


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(n_classes, pretrained=True):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = mobilenet_v3_small(weights=weights)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


class LeafDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.imgs, self.labels, self.tf = [load_rgb(p) for p in paths], labels, tf  # preloaded (small dataset)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), int(self.labels[i])


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    return torch.cat([torch.softmax(model(x.to(device)), 1).cpu() for x, _ in loader]).numpy()


def run_epoch(model, loader, device, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    if train:
        model.features.eval()  # frozen backbone: keep BatchNorm statistics fixed
    tot, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot / n, correct / n


def main():
    cfg = load_config()
    mc = cfg["mobilenet"]
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = get_device()
    df = load_metadata()
    classes = list(cfg["known"])
    tr, va = df[df.split == "train"], df[df.split == "val"]
    assert (tr.role == "known").all(), "unknown species must never be in training"
    sz = cfg["image_size_cnn"]
    mk = lambda d, tf, shuffle: DataLoader(LeafDS(d.path, d.label.values, tf), batch_size=mc["batch_size"], shuffle=shuffle)
    g = torch.Generator().manual_seed(cfg["seed"])
    tr_loader = DataLoader(LeafDS(tr.path, tr.label.values, train_transform(sz)), batch_size=mc["batch_size"], shuffle=True, generator=g)
    va_loader = mk(va, eval_transform(sz), False)

    model = build_model(len(classes)).to(device)
    for p in model.features.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=mc["lr"])
    loss_fn = nn.CrossEntropyLoss()

    hist, best, bad, t0 = [], float("inf"), 0, time.time()
    Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
    for ep in range(1, mc["epochs"] + 1):
        trl, tra = run_epoch(model, tr_loader, device, loss_fn, opt)
        vll, vla = run_epoch(model, va_loader, device, loss_fn)
        hist.append(dict(epoch=ep, train_loss=trl, train_acc=tra, val_loss=vll, val_acc=vla))
        print(f"epoch {ep:2d} train_loss {trl:.4f} acc {tra:.3f} | val_loss {vll:.4f} acc {vla:.3f}")
        if vll < best:
            best, bad = vll, 0
            torch.save({"state_dict": model.state_dict(), "classes": classes, "image_size": sz, "epoch": ep},
                       "artifacts/checkpoints/mobilenet_v3_small.pth")
        else:
            bad += 1
            if bad >= mc["patience"]:
                print("early stopping"); break
    train_s = time.time() - t0

    ck = torch.load("artifacts/checkpoints/mobilenet_v3_small.pth", map_location=device)
    model.load_state_dict(ck["state_dict"])
    out, timing = {}, {}
    for name, split in (("val", "val"), ("test", "test"), ("unknown", "unknown_test")):
        sub = df[df.split == split]
        dl = mk(sub, eval_transform(sz), False)
        t = time.time()
        out[f"{name}_probs"] = predict_probs(model, dl, device)
        timing[name] = (time.time() - t) / len(sub)  # excludes image decode (preloaded)
    np.savez("artifacts/predictions/mobilenet.npz", **out)
    info = {"model": "MobileNetV3-Small", "weights": "IMAGENET1K_V1 (pretrained, ImageNet)", "device": str(device),
            "frozen": "features (backbone); trained: classifier head", "best_epoch": ck["epoch"], "selection_metric": "val_loss",
            "best_val_loss": best, "train_seconds": round(train_s, 1), "config": mc, "image_size": sz, "seed": cfg["seed"],
            "inference_seconds_per_image_excl_decode": {k: round(v, 5) for k, v in timing.items()},
            "torch": torch.__version__, "torchvision": torchvision.__version__, "python": platform.python_version(), "history": hist}
    json.dump(info, open("artifacts/logs/mobilenet_train.json", "w"), indent=2)
    print("best epoch", ck["epoch"], "device", device, "train s", round(train_s, 1))


if __name__ == "__main__":
    main()
