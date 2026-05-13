# -*- coding: utf-8 -*-
"""
生成模型实验平台：AE / VAE / DCGAN / Diffusers
面向课堂展示的 Streamlit 版本

运行：
streamlit run app.py
"""

import io
import os
import re
import zipfile
import random
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import plotly.express as px

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid

warnings.filterwarnings("ignore")

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except Exception:
    StableDiffusionPipeline = None
    DIFFUSERS_AVAILABLE = False


# =========================
# 页面基础配置
# =========================
st.set_page_config(
    page_title="生成模型实验平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 28
INPUT_DIM = IMG_SIZE * IMG_SIZE
DATA_ROOT = "./data"

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# =========================
# 中文字体与页面样式
# =========================
def setup_chinese_font() -> str:
    """
    尽量自动寻找可用中文字体，避免 matplotlib 图片标题出现口口口。
    在 Windows / macOS / Linux / Streamlit Cloud 上都做了兼容。
    """
    preferred_names = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "Source Han Sans CN",
        "DejaVu Sans",
    ]

    installed = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in preferred_names:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            matplotlib.rcParams["font.family"] = "sans-serif"
            return name

    # 尝试常见字体文件路径
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            name = prop.get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            matplotlib.rcParams["font.family"] = "sans-serif"
            return name

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


FONT_NAME = setup_chinese_font()

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1420px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: .25rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        border: 1px solid rgba(148,163,184,.28);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(255,255,255,.86), rgba(248,250,252,.86));
        box-shadow: 0 8px 24px rgba(15,23,42,.06);
    }
    .soft-card {
        border: 1px solid rgba(148,163,184,.25);
        border-radius: 20px;
        padding: 1.1rem 1.2rem;
        background: rgba(248,250,252,.70);
        margin-bottom: 1rem;
    }
    .hint {
        color: #64748b;
        font-size: .92rem;
    }
    div[data-testid="stTabs"] button {
        font-size: .98rem;
        font-weight: 650;
    }
    .stButton>button {
        border-radius: 14px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 通用工具
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainHistory:
    train_total: List[float]
    val_total: List[float]
    train_recon: List[float]
    val_recon: List[float]
    train_kld: List[float]
    val_kld: List[float]


def init_state():
    defaults = {
        "ae_model": None,
        "vae_model": None,
        "ae_history": None,
        "vae_history": None,
        "samples": None,
        "latent_full": None,
        "latent_labels": None,
        "label_names": None,
        "data_info": None,
        "gan_G": None,
        "gan_D": None,
        "gan_history": None,
        "gan_last_noise": None,
        "gan_last_scores": None,
        "gan_last_gen_imgs": None,
        "diff_images": [],
        "diff_meta": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def to_np_img(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy()


def mse_value(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def fig_to_streamlit(fig, width=True):
    st.pyplot(fig, use_container_width=width)
    plt.close(fig)


# =========================
# 数据集
# =========================
class UploadedImageDataset(Dataset):
    def __init__(self, images: List[torch.Tensor], labels: List[int], label_names: List[str]):
        self.images = images
        self.labels = labels
        self.label_names = label_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def infer_label_from_name(path: str) -> str:
    norm_path = path.replace("\\", "/")
    parts = [p for p in norm_path.split("/") if p and p != "__MACOSX"]

    if len(parts) >= 2:
        parent = parts[-2].strip()
        if parent and not parent.startswith("."):
            return parent

    base = os.path.basename(norm_path)
    stem = os.path.splitext(base)[0].strip()

    for sep in ["_", "-", "."]:
        if sep in stem:
            prefix = stem.split(sep)[0].strip()
            if prefix:
                return prefix

    return "未标注"


def pil_to_tensor_28(img: Image.Image, invert: bool = False, keep_aspect: bool = True) -> torch.Tensor:
    img = img.convert("L")

    if keep_aspect:
        img.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
        x = (IMG_SIZE - img.width) // 2
        y = (IMG_SIZE - img.height) // 2
        canvas.paste(img, (x, y))
        img = canvas
    else:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    if invert:
        img = ImageOps.invert(img)

    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


@st.cache_data(show_spinner=False)
def load_custom_dataset_from_bytes(
    image_payloads: Tuple[Tuple[str, bytes], ...],
    zip_payload: Optional[bytes],
    invert: bool,
    keep_aspect: bool,
):
    items = []

    if zip_payload is not None:
        with zipfile.ZipFile(io.BytesIO(zip_payload), "r") as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    with zf.open(name) as f:
                        img = Image.open(io.BytesIO(f.read()))
                        tensor = pil_to_tensor_28(img, invert=invert, keep_aspect=keep_aspect)
                        label = infer_label_from_name(name)
                        items.append((tensor, label))

    for filename, raw in image_payloads:
        img = Image.open(io.BytesIO(raw))
        tensor = pil_to_tensor_28(img, invert=invert, keep_aspect=keep_aspect)
        label = infer_label_from_name(filename)
        items.append((tensor, label))

    if not items:
        raise ValueError("没有读取到图片。请上传图片或 ZIP 图片包。")

    unique_labels = sorted(set(label for _, label in items))
    label_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
    images = [img for img, _ in items]
    labels = [label_to_idx[label] for _, label in items]

    return UploadedImageDataset(images, labels, unique_labels)


@st.cache_data(show_spinner=False)
def get_builtin_datasets(dataset_name: str):
    transform = transforms.ToTensor()
    if dataset_name == "MNIST":
        train_ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
        label_names = [str(i) for i in range(10)]
    else:
        train_ds = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
        label_names = FASHION_LABELS
    return train_ds, test_ds, label_names


def build_loaders_builtin(train_ds, test_ds, batch_size: int, train_limit: int, test_limit: int, seed: int):
    if train_limit and train_limit < len(train_ds):
        train_ds = Subset(train_ds, list(range(train_limit)))
    if test_limit and test_limit < len(test_ds):
        test_ds = Subset(test_ds, list(range(test_limit)))

    val_size = max(1, int(0.1 * len(train_ds)))
    train_size = len(train_ds) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    info = {
        "source": "MNIST" if isinstance(getattr(train_ds, "dataset", train_ds), datasets.MNIST) else "FashionMNIST",
        "train_count": len(train_subset),
        "val_count": len(val_subset),
        "test_count": len(test_ds),
    }
    return train_loader, val_loader, test_loader, info


def build_loaders_custom(dataset: UploadedImageDataset, batch_size: int, seed: int):
    n = len(dataset)
    if n < 10:
        raise ValueError("自定义数据集至少需要 10 张图片，建议 30 张以上。")

    test_size = max(1, int(0.1 * n))
    val_size = max(1, int(0.1 * n))
    train_size = n - val_size - test_size
    if train_size < 1:
        raise ValueError("图片数量太少，无法切分训练集、验证集和测试集。")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    info = {
        "source": "自定义上传图片",
        "train_count": len(train_subset),
        "val_count": len(val_subset),
        "test_count": len(test_subset),
    }
    return train_loader, val_loader, test_loader, info


def prepare_data(
    source_mode: str,
    dataset_name: str,
    uploaded_images,
    uploaded_zip,
    invert_custom: bool,
    keep_aspect: bool,
    batch_size: int,
    train_limit: int,
    test_limit: int,
    seed: int,
):
    if source_mode == "内置数据集":
        train_ds, test_ds, label_names = get_builtin_datasets(dataset_name)
        train_loader, val_loader, test_loader, info = build_loaders_builtin(
            train_ds, test_ds, batch_size, train_limit, test_limit, seed
        )
        info["source"] = dataset_name
        return train_loader, val_loader, test_loader, label_names, info

    image_payloads = tuple((f.name, f.getvalue()) for f in (uploaded_images or []))
    zip_payload = uploaded_zip.getvalue() if uploaded_zip is not None else None
    dataset = load_custom_dataset_from_bytes(image_payloads, zip_payload, invert_custom, keep_aspect)
    train_loader, val_loader, test_loader, info = build_loaders_custom(dataset, batch_size, seed)
    return train_loader, val_loader, test_loader, dataset.label_names, info


# =========================
# AE / VAE
# =========================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, INPUT_DIM), nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        recon = self.decoder(z).view(-1, 1, IMG_SIZE, IMG_SIZE)
        return recon, z


class SimpleVAE(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(INPUT_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, INPUT_DIM)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z).view(-1, 1, IMG_SIZE, IMG_SIZE)
        return recon, mu, logvar, z


def ae_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction="mean")


def vae_loss(recon_x, x, mu, logvar, beta: float):
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld


def train_autoencoder(model, train_loader, val_loader, epochs: int, lr: float, progress_callback=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory([], [], [], [], [], [])

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = ae_loss(recon, x)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                recon, _ = model(x)
                val_losses.append(ae_loss(recon, x).item())

        train_mean = float(np.mean(train_losses))
        val_mean = float(np.mean(val_losses))
        history.train_total.append(train_mean)
        history.val_total.append(val_mean)
        history.train_recon.append(train_mean)
        history.val_recon.append(val_mean)
        history.train_kld.append(0.0)
        history.val_kld.append(0.0)

        if progress_callback:
            progress_callback(epoch + 1, epochs, f"AE 训练中：{epoch + 1}/{epochs}")

    return history


def train_vae(model, train_loader, val_loader, epochs: int, lr: float, beta: float, progress_callback=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory([], [], [], [], [], [])

    for epoch in range(epochs):
        model.train()
        train_total, train_recon, train_kld = [], [], []

        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(x)
            total, recon_loss, kld = vae_loss(recon, x, mu, logvar, beta)
            total.backward()
            optimizer.step()
            train_total.append(total.item())
            train_recon.append(recon_loss.item())
            train_kld.append(kld.item())

        model.eval()
        val_total, val_recon, val_kld = [], [], []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                recon, mu, logvar, _ = model(x)
                total, recon_loss, kld = vae_loss(recon, x, mu, logvar, beta)
                val_total.append(total.item())
                val_recon.append(recon_loss.item())
                val_kld.append(kld.item())

        history.train_total.append(float(np.mean(train_total)))
        history.val_total.append(float(np.mean(val_total)))
        history.train_recon.append(float(np.mean(train_recon)))
        history.val_recon.append(float(np.mean(val_recon)))
        history.train_kld.append(float(np.mean(train_kld)))
        history.val_kld.append(float(np.mean(val_kld)))

        if progress_callback:
            progress_callback(epoch + 1, epochs, f"VAE 训练中：{epoch + 1}/{epochs}")

    return history


@torch.no_grad()
def collect_test_examples(ae_model, vae_model, test_loader, max_items: int = 128):
    ae_model.eval()
    vae_model.eval()

    originals, ae_recons, vae_recons, labels = [], [], [], []
    count = 0

    for x, y in test_loader:
        x = x.to(DEVICE)
        ae_out, _ = ae_model(x)
        vae_out, _, _, _ = vae_model(x)
        originals.append(x.cpu())
        ae_recons.append(ae_out.cpu())
        vae_recons.append(vae_out.cpu())
        labels.append(y.cpu())
        count += x.size(0)
        if count >= max_items:
            break

    return (
        torch.cat(originals, dim=0)[:max_items],
        torch.cat(ae_recons, dim=0)[:max_items],
        torch.cat(vae_recons, dim=0)[:max_items],
        torch.cat(labels, dim=0)[:max_items],
    )


@torch.no_grad()
def collect_vae_latent(vae_model, data_loader):
    vae_model.eval()
    mus, labels = [], []
    for x, y in data_loader:
        x = x.to(DEVICE)
        mu, _ = vae_model.encode(x.view(x.size(0), -1))
        mus.append(mu.cpu())
        labels.append(y.cpu())
    return torch.cat(mus).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def generate_from_latent(vae_model, z: np.ndarray):
    vae_model.eval()
    z_tensor = torch.tensor(z, dtype=torch.float32).to(DEVICE).unsqueeze(0)
    return vae_model.decode(z_tensor).view(IMG_SIZE, IMG_SIZE).cpu().numpy()


def interpolate(z1, z2, steps: int):
    return [z1 * (1 - a) + z2 * a for a in np.linspace(0, 1, steps)]


# =========================
# DCGAN
# =========================
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, IMG_SIZE * IMG_SIZE),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 1, IMG_SIZE, IMG_SIZE)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def train_dcgan(train_loader, epochs: int, latent_dim: int, lr: float, progress_callback=None):
    G = Generator(latent_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        epoch_d, epoch_g = [], []

        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(DEVICE) * 2 - 1
            bs = real_imgs.size(0)

            real_labels = torch.ones(bs, 1, device=DEVICE)
            fake_labels = torch.zeros(bs, 1, device=DEVICE)

            opt_d.zero_grad()
            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_imgs = G(z)
            real_loss = criterion(D(real_imgs), real_labels)
            fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_imgs = G(z)
            g_loss = criterion(D(fake_imgs), real_labels)
            g_loss.backward()
            opt_g.step()

            epoch_d.append(d_loss.item())
            epoch_g.append(g_loss.item())

        d_losses.append(float(np.mean(epoch_d)))
        g_losses.append(float(np.mean(epoch_g)))

        if progress_callback:
            progress_callback(
                epoch + 1,
                epochs,
                f"DCGAN 训练中：{epoch + 1}/{epochs}  |  D={d_losses[-1]:.4f}  G={g_losses[-1]:.4f}",
            )

    return G, D, {"d_losses": d_losses, "g_losses": g_losses, "latent_dim": latent_dim}


@torch.no_grad()
def sample_dcgan(G, D, latent_dim: int, n: int = 16, seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    G.eval()
    D.eval()
    z = torch.randn(n, latent_dim, device=DEVICE)
    gen_imgs = G(z)
    scores = D(gen_imgs).view(-1).cpu().numpy()
    return z.cpu().numpy(), gen_imgs.cpu(), scores


# =========================
# 可视化
# =========================
def build_recon_figure(original, ae_rec, vae_rec):
    err_ae = np.abs(original - ae_rec)
    err_vae = np.abs(original - vae_rec)

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.7))
    titles = [
        "输入图像", "AE 重构", "AE 误差热力图",
        "输入图像", "VAE 重构", "VAE 误差热力图",
    ]
    imgs = [original, ae_rec, err_ae, original, vae_rec, err_vae]
    cmaps = ["gray", "gray", "hot", "gray", "gray", "hot"]

    for ax, img, title, cmap in zip(axes.ravel(), imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    return fig


def build_loss_figure(ae_hist, vae_hist):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(ae_hist.train_total, label="AE 训练")
    axes[0].plot(ae_hist.val_total, label="AE 验证")
    axes[0].plot(vae_hist.train_total, label="VAE 训练")
    axes[0].plot(vae_hist.val_total, label="VAE 验证")
    axes[0].set_title("总损失曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(vae_hist.train_recon, label="重构误差（训练）")
    axes[1].plot(vae_hist.val_recon, label="重构误差（验证）")
    axes[1].plot(vae_hist.train_kld, label="KL 散度（训练）")
    axes[1].plot(vae_hist.val_kld, label="KL 散度（验证）")
    axes[1].set_title("VAE 损失组成")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig


def build_interpolation_figure(imgs):
    fig, axes = plt.subplots(1, len(imgs), figsize=(1.8 * len(imgs), 2.2))
    if len(imgs) == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{i + 1}", fontsize=11)
        ax.axis("off")
    fig.suptitle("潜空间插值序列", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def build_gan_loss_figure(d_losses, g_losses):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(d_losses, label="判别器损失 D")
    ax.plot(g_losses, label="生成器损失 G")
    ax.set_title("DCGAN 训练损失曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def build_generated_grid_figure(gen_imgs, title="DCGAN 生成样本网格"):
    grid = make_grid(gen_imgs, nrow=4, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).squeeze().numpy()

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(grid_np, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def show_metrics_row(metrics: Dict[str, str]):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="hint">{label}</div>
                    <div style="font-size:1.45rem;font-weight:800;margin-top:.2rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================
# Diffusers
# =========================
@st.cache_resource(show_spinner=False)
def load_diffusion_pipeline(model_id: str):
    if StableDiffusionPipeline is None:
        raise RuntimeError("当前环境未安装 diffusers。")
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(DEVICE)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def build_placeholder_diffusion(prompt, negative_prompt, seed, steps, guidance_scale):
    """
    无法加载 diffusers 时，用确定性的伪图像做参数实验占位。
    这样课堂展示时即使云端资源不足，也不会出现空白页面。
    """
    rng_seed = abs(hash((prompt, negative_prompt, seed, steps, round(guidance_scale, 2)))) % (2**32)
    rng = np.random.default_rng(rng_seed)
    base = rng.normal(0.5, 0.16, size=(256, 256))
    yy, xx = np.mgrid[0:256, 0:256]
    blob_count = 5 + int(guidance_scale) % 5
    for _ in range(blob_count):
        cx, cy = rng.integers(20, 236, size=2)
        sigma = rng.uniform(12, 42)
        amp = rng.uniform(-0.35, 0.45)
        base += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    base = np.clip(base, 0, 1)
    arr = (base * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


# =========================
# 主应用
# =========================
def main():
    init_state()

    st.markdown('<div class="main-title">🧠 生成模型实验平台</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">AE / VAE 重构对比、VAE 潜空间交互、DCGAN 生成实验、Diffusers 文本到图像参数对比</div>',
        unsafe_allow_html=True,
    )

    show_metrics_row({
        "运行设备": str(DEVICE).upper(),
        "图像尺寸": "28 × 28",
        "中文字体": FONT_NAME,
        "模型模块": "AE / VAE / GAN / Diffusion",
    })

    with st.sidebar:
        st.header("实验配置")

        source_mode = st.radio("数据来源", ["内置数据集", "自定义上传图片"])
        dataset_name = "MNIST"
        uploaded_images = None
        uploaded_zip = None
        invert_custom = False
        keep_aspect = True

        if source_mode == "内置数据集":
            dataset_name = st.selectbox("数据集", ["MNIST", "FashionMNIST"])
        else:
            uploaded_images = st.file_uploader(
                "上传图片",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                accept_multiple_files=True,
            )
            uploaded_zip = st.file_uploader("上传 ZIP 图片包", type=["zip"])
            invert_custom = st.checkbox("反色预处理", value=False)
            keep_aspect = st.checkbox("保持宽高比并居中", value=True)

        st.divider()
        seed = st.number_input("随机种子", 0, 999999, 42, 1)
        batch_size = st.number_input("Batch Size", 8, 512, 128, 8)

        if source_mode == "内置数据集":
            train_limit = st.number_input("训练样本上限", 100, 60000, 10000, 100)
            test_limit = st.number_input("测试样本上限", 32, 10000, 1000, 32)
        else:
            train_limit, test_limit = 0, 0

        st.divider()
        st.subheader("AE / VAE")
        aevae_epochs = st.number_input("训练轮数", 1, 300, 8, 1)
        lr = st.number_input("学习率", 0.00001, 1.0, 0.001, 0.0001, format="%.5f")
        ae_latent_dim = st.number_input("AE 潜变量维度", 2, 128, 16, 1)
        vae_latent_dim = st.number_input("VAE 潜变量维度", 2, 32, 2, 1)
        beta = st.slider("VAE Beta", 0.0, 10.0, 1.0, 0.1)

        st.divider()
        st.subheader("DCGAN")
        gan_epochs = st.number_input("GAN 训练轮数", 1, 300, 30, 1)
        gan_lr = st.number_input("GAN 学习率", 0.00001, 1.0, 0.0002, 0.0001, format="%.5f")
        gan_latent_dim = st.number_input("噪声维度", 8, 256, 100, 4)

    tabs = st.tabs([
        "① AE / VAE 重构",
        "② VAE 潜空间",
        "③ DCGAN 生成",
        "④ 文本到图像参数实验",
    ])

    # --------- Tab 1 ---------
    with tabs[0]:
        st.markdown("### AE 与 VAE 重构对比")

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(
                """
                <div class="soft-card">
                本页训练普通自编码器和简化 VAE，并展示输入图像、两种重构结果、误差热力图与 loss 曲线。
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            start_aevae = st.button("开始训练 AE + VAE", type="primary", use_container_width=True)

        if start_aevae:
            try:
                set_seed(int(seed))
                train_loader, val_loader, test_loader, label_names, info = prepare_data(
                    source_mode, dataset_name, uploaded_images, uploaded_zip,
                    bool(invert_custom), bool(keep_aspect), int(batch_size),
                    int(train_limit), int(test_limit), int(seed)
                )

                ae_model = Autoencoder(int(ae_latent_dim)).to(DEVICE)
                vae_model = SimpleVAE(int(vae_latent_dim)).to(DEVICE)

                st.session_state.label_names = label_names
                st.session_state.data_info = info

                status = st.empty()
                progress = st.progress(0)

                def ae_cb(epoch, total, msg):
                    progress.progress(int(50 * epoch / total))
                    status.info(msg)

                def vae_cb(epoch, total, msg):
                    progress.progress(50 + int(50 * epoch / total))
                    status.info(msg)

                ae_history = train_autoencoder(ae_model, train_loader, val_loader, int(aevae_epochs), float(lr), ae_cb)
                vae_history = train_vae(vae_model, train_loader, val_loader, int(aevae_epochs), float(lr), float(beta), vae_cb)

                max_items = min(256, info["test_count"])
                samples = collect_test_examples(ae_model, vae_model, test_loader, max_items=max_items)
                latent_full, latent_labels = collect_vae_latent(vae_model, test_loader)

                st.session_state.ae_model = ae_model
                st.session_state.vae_model = vae_model
                st.session_state.ae_history = ae_history
                st.session_state.vae_history = vae_history
                st.session_state.samples = samples
                st.session_state.latent_full = latent_full
                st.session_state.latent_labels = latent_labels

                progress.progress(100)
                status.success("训练完成，可以查看重构结果和潜空间。")
            except Exception as e:
                st.error(f"训练失败：{e}")

        if st.session_state.samples is not None:
            originals, ae_recons, vae_recons, labels = st.session_state.samples
            st.markdown("#### 重构结果")

            top_cols = st.columns([2, 1])
            with top_cols[0]:
                sample_idx = st.slider("选择测试样本", 0, len(originals) - 1, 0, 1)
            with top_cols[1]:
                label_id = int(labels[sample_idx].item())
                label_name = (
                    st.session_state.label_names[label_id]
                    if st.session_state.label_names and label_id < len(st.session_state.label_names)
                    else str(label_id)
                )
                st.metric("样本标签", label_name)

            original = to_np_img(originals[sample_idx])
            ae_rec = to_np_img(ae_recons[sample_idx])
            vae_rec = to_np_img(vae_recons[sample_idx])
            ae_mse = mse_value(original, ae_rec)
            vae_mse = mse_value(original, vae_rec)

            fig = build_recon_figure(original, ae_rec, vae_rec)
            fig_to_streamlit(fig, width=False)

            show_metrics_row({
                "AE 重构 MSE": f"{ae_mse:.6f}",
                "VAE 重构 MSE": f"{vae_mse:.6f}",
                "当前更优": "AE" if ae_mse < vae_mse else "VAE",
                "测试样本数": str(len(originals)),
            })

            st.markdown("#### Loss 曲线")
            fig_loss = build_loss_figure(st.session_state.ae_history, st.session_state.vae_history)
            fig_to_streamlit(fig_loss)

            if st.session_state.data_info:
                info = st.session_state.data_info
                st.caption(
                    f"数据来源：{info['source']} ｜ 训练集：{info['train_count']} ｜ 验证集：{info['val_count']} ｜ 测试集：{info['test_count']}"
                )
        else:
            st.info("点击“开始训练 AE + VAE”后展示结果。")

    # --------- Tab 2 ---------
    with tabs[1]:
        st.markdown("### VAE 二维潜空间交互")

        if st.session_state.vae_model is None or st.session_state.latent_full is None:
            st.info("请先在第 ① 页训练 AE + VAE。")
        else:
            latent_full = st.session_state.latent_full
            latent_labels = st.session_state.latent_labels
            label_names = st.session_state.label_names or []

            z1_all, z2_all = latent_full[:, 0], latent_full[:, 1]
            label_texts = [
                label_names[int(lb)] if int(lb) < len(label_names) else str(int(lb))
                for lb in latent_labels
            ]

            df = pd.DataFrame({
                "样本编号": np.arange(len(latent_full)),
                "z1": z1_all,
                "z2": z2_all,
                "类别编号": latent_labels.astype(int),
                "类别": label_texts,
            })

            color_mode = st.radio("着色方式", ["按类别", "按样本编号"], horizontal=True)
            if color_mode == "按类别":
                fig_scatter = px.scatter(
                    df, x="z1", y="z2", color="类别",
                    hover_data=["样本编号", "类别编号"],
                    title="VAE 潜空间散点图"
                )
            else:
                fig_scatter = px.scatter(
                    df, x="z1", y="z2", color="样本编号",
                    hover_data=["类别", "类别编号"],
                    title="VAE 潜空间散点图"
                )
            fig_scatter.update_layout(height=520, legend_title_text="图例")
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("#### 按坐标生成图像")
            c1, c2, c3 = st.columns([1, 1, 1])
            z1_input = c1.number_input("z1", value=float(np.mean(z1_all)), format="%.4f")
            z2_input = c2.number_input("z2", value=float(np.mean(z2_all)), format="%.4f")
            gen_btn = c3.button("生成图像", type="primary", use_container_width=True)

            if gen_btn:
                latent_dim = st.session_state.vae_model.latent_dim
                z = np.zeros(latent_dim, dtype=np.float32)
                z[0], z[1] = float(z1_input), float(z2_input)
                img = generate_from_latent(st.session_state.vae_model, z)
                fig_gen, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img, cmap="gray")
                ax.set_title("坐标解码结果")
                ax.axis("off")
                fig_to_streamlit(fig_gen, width=False)

            st.markdown("#### 样本插值")
            c1, c2, c3 = st.columns(3)
            idx1 = c1.number_input("样本 A", min_value=0, max_value=len(latent_full) - 1, value=0, step=1)
            idx2 = c2.number_input("样本 B", min_value=0, max_value=len(latent_full) - 1, value=min(1, len(latent_full)-1), step=1)
            steps = c3.slider("插值步数", 3, 12, 7, 1)

            if st.button("生成插值序列", use_container_width=True):
                seq = interpolate(latent_full[int(idx1)], latent_full[int(idx2)], int(steps))
                imgs = [generate_from_latent(st.session_state.vae_model, z) for z in seq]
                fig_interp = build_interpolation_figure(imgs)
                fig_to_streamlit(fig_interp)

    # --------- Tab 3 ---------
    with tabs[2]:
        st.markdown("### DCGAN 生成实验")

        left, right = st.columns(2)
        train_gan_btn = left.button("开始训练 DCGAN", type="primary", use_container_width=True)
        sample_seed = right.number_input("生成样本 seed", 0, 999999, int(seed), 1)
        sample_gan_btn = right.button("生成样本网格", use_container_width=True)

        if train_gan_btn:
            try:
                set_seed(int(seed))
                train_loader, _, _, _, info = prepare_data(
                    source_mode, dataset_name, uploaded_images, uploaded_zip,
                    bool(invert_custom), bool(keep_aspect), int(batch_size),
                    int(train_limit), int(test_limit), int(seed)
                )

                status = st.empty()
                progress = st.progress(0)

                def gan_cb(epoch, total, msg):
                    progress.progress(int(100 * epoch / total))
                    status.info(msg)

                G, D, history = train_dcgan(
                    train_loader, int(gan_epochs), int(gan_latent_dim), float(gan_lr), gan_cb
                )

                st.session_state.gan_G = G
                st.session_state.gan_D = D
                st.session_state.gan_history = history
                st.session_state.data_info = info

                progress.progress(100)
                status.success("DCGAN 训练完成。")
            except Exception as e:
                st.error(f"DCGAN 训练失败：{e}")

        if st.session_state.gan_history is not None:
            fig_gan_loss = build_gan_loss_figure(
                st.session_state.gan_history["d_losses"],
                st.session_state.gan_history["g_losses"],
            )
            fig_to_streamlit(fig_gan_loss)

        if sample_gan_btn:
            if st.session_state.gan_G is None or st.session_state.gan_D is None:
                st.warning("请先训练 DCGAN。")
            else:
                z, gen_imgs, scores = sample_dcgan(
                    st.session_state.gan_G,
                    st.session_state.gan_D,
                    int(st.session_state.gan_history["latent_dim"]),
                    n=16,
                    seed=int(sample_seed),
                )
                st.session_state.gan_last_noise = z
                st.session_state.gan_last_gen_imgs = gen_imgs
                st.session_state.gan_last_scores = scores

        if st.session_state.gan_last_gen_imgs is not None:
            c1, c2 = st.columns([1.1, 1])
            with c1:
                fig_grid = build_generated_grid_figure(st.session_state.gan_last_gen_imgs)
                fig_to_streamlit(fig_grid, width=False)
            with c2:
                st.markdown("#### 生成器输入噪声")
                st.code(np.array2string(st.session_state.gan_last_noise[0][:16], precision=4), language="text")

                st.markdown("#### 判别器分数")
                score_df = pd.DataFrame({
                    "样本": [f"样本 {i+1}" for i in range(len(st.session_state.gan_last_scores))],
                    "判别器分数": st.session_state.gan_last_scores,
                })
                st.dataframe(score_df, use_container_width=True, hide_index=True)
                st.caption("分数越接近 1，表示判别器越倾向于判断该生成图像为真实样本。")

    # --------- Tab 4 ---------
    with tabs[3]:
        st.markdown("### Diffusers 文本到图像参数实验")

        st.markdown(
            """
            <div class="soft-card">
            可调整 prompt、negative prompt、采样步数、seed 和 guidance scale，并对比不同参数下的生成结果。
            如果当前环境无法加载扩散模型，页面会自动使用轻量占位生成，保证参数实验仍可展示。
            </div>
            """,
            unsafe_allow_html=True,
        )

        use_real_diffusion = st.toggle("使用 diffusers 模型生成", value=False, disabled=not DIFFUSERS_AVAILABLE)
        model_id = st.text_input("模型名称", value="hf-internal-testing/tiny-stable-diffusion-pipe")

        c1, c2 = st.columns(2)
        prompt = c1.text_area("Prompt", value="a cute black and white fashion item, clean background", height=100)
        negative_prompt = c2.text_area("Negative Prompt", value="blurry, low quality, distorted", height=100)

        c1, c2, c3 = st.columns(3)
        steps = c1.slider("采样步数", 5, 50, 20, 1)
        guidance_scale = c2.slider("Guidance Scale", 1.0, 15.0, 7.5, 0.5)
        diff_seed = c3.number_input("Seed", 0, 999999, 42, 1)

        cols = st.columns(2)
        gen_diff_btn = cols[0].button("生成并加入对比", type="primary", use_container_width=True)
        clear_diff_btn = cols[1].button("清空对比结果", use_container_width=True)

        if clear_diff_btn:
            st.session_state.diff_images = []
            st.session_state.diff_meta = []

        if gen_diff_btn:
            try:
                if use_real_diffusion:
                    pipe = load_diffusion_pipeline(model_id)
                    generator = torch.Generator(device=str(DEVICE)).manual_seed(int(diff_seed))
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt.strip() else None,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance_scale),
                        generator=generator,
                    )
                    image = result.images[0]
                    mode = "Diffusers"
                else:
                    image = build_placeholder_diffusion(prompt, negative_prompt, int(diff_seed), int(steps), float(guidance_scale))
                    mode = "轻量占位生成"

                st.session_state.diff_images.append(image)
                st.session_state.diff_meta.append({
                    "模式": mode,
                    "Seed": int(diff_seed),
                    "Steps": int(steps),
                    "Guidance": float(guidance_scale),
                    "Prompt": prompt,
                    "Negative": negative_prompt,
                })
                st.success("已加入对比结果。")
            except Exception as e:
                st.error(f"生成失败：{e}")

        if st.session_state.diff_images:
            st.markdown("#### 参数对比结果")
            cols = st.columns(min(3, len(st.session_state.diff_images)))
            for i, (img, meta) in enumerate(zip(st.session_state.diff_images, st.session_state.diff_meta)):
                with cols[i % len(cols)]:
                    st.image(img, use_container_width=True)
                    st.caption(
                        f"{meta['模式']} ｜ seed={meta['Seed']} ｜ steps={meta['Steps']} ｜ guidance={meta['Guidance']}"
                    )
        else:
            st.info("点击“生成并加入对比”后展示结果。")


if __name__ == "__main__":
    main()
