import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from hydra.utils.io import ensure_dir, extract_lanl_window, get_earliest_time_in_gz, infer_delim_from_peek
from hydra.utils.metrics import print_header, save_json

LANL_AUTH_COLS = [
    "time",
    "src_user",
    "dst_user",
    "src_comp",
    "dst_comp",
    "auth_type",
    "logon_type",
    "auth_orient",
    "success",
]
LANL_RT_COLS = ["time", "user", "src_comp", "dst_comp"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scan_lanl_window(
    auth_gz: str,
    redteam_gz: str,
    out_dir: str,
    window_seconds: int = 7200,
    max_auth_lines: Optional[int] = 2_000_000,
) -> None:
    ensure_dir(out_dir)

    print_header("LANL scan: peek + infer delimiter")
    delim = infer_delim_from_peek(auth_gz)
    print(f"Inferred delimiter: {repr(delim)}")

    rt_delim = infer_delim_from_peek(redteam_gz)
    if rt_delim != delim:
        print(f"Redteam delimiter differs: {repr(rt_delim)}")

    tmin_rt = get_earliest_time_in_gz(redteam_gz, delim=rt_delim)
    t0, t1 = tmin_rt, tmin_rt + window_seconds
    print(f"Earliest redteam time: {tmin_rt}")
    print(f"Extracting window: [{t0}, {t1}] ({window_seconds}s)")

    auth_out = f"{out_dir}/auth_window.csv"
    rt_out = f"{out_dir}/redteam_window.csv"

    na = extract_lanl_window(auth_gz, auth_out, t0, t1, delim=delim, max_lines=max_auth_lines)
    nr = extract_lanl_window(redteam_gz, rt_out, t0, t1, delim=rt_delim, max_lines=None)
    print(f"Wrote auth window lines: {na:,}")
    print(f"Wrote redteam window lines: {nr:,}")

    save_json(
        {
            "t0": t0,
            "t1": t1,
            "window_seconds": window_seconds,
            "auth_window_lines": na,
            "redteam_window_lines": nr,
            "auth_window_path": auth_out,
            "redteam_window_path": rt_out,
        },
        f"{out_dir}/lanl_scan.json",
    )


def label_auth_by_redteam(df_auth: pd.DataFrame, df_rt: pd.DataFrame, tol: int = 0) -> pd.DataFrame:
    df_auth = df_auth.copy()
    df_rt = df_rt.copy()
    df_auth["time"] = df_auth["time"].astype(int)
    df_rt["time"] = df_rt["time"].astype(int)

    if tol == 0:
        df_rt_keyed = df_rt.drop_duplicates(subset=["time", "user", "src_comp", "dst_comp"])
        df_auth2 = df_auth.rename(columns={"src_user": "user"})
        merged = df_auth2.merge(
            df_rt_keyed.assign(redteam=1), on=["time", "user", "src_comp", "dst_comp"], how="left"
        )
        merged["is_malicious"] = merged["redteam"].fillna(0).astype(int)
        return merged.drop(columns=["redteam"])

    rows = []
    for _, r in df_rt.iterrows():
        for tt in range(r["time"] - tol, r["time"] + tol + 1):
            rows.append((tt, r["user"], r["src_comp"], r["dst_comp"]))
    exp = pd.DataFrame(rows, columns=["time", "user", "src_comp", "dst_comp"]).drop_duplicates()
    df_auth2 = df_auth.rename(columns={"src_user": "user"})
    merged = df_auth2.merge(exp.assign(redteam=1), on=["time", "user", "src_comp", "dst_comp"], how="left")
    merged["is_malicious"] = merged["redteam"].fillna(0).astype(int)
    return merged.drop(columns=["redteam"])


def build_user_dest_sequences(df_auth: pd.DataFrame) -> Dict[str, List[str]]:
    df = df_auth.sort_values("time")
    seqs = {}
    for user, g in df.groupby("user"):
        seqs[user] = g["dst_comp"].astype(str).tolist()
    return seqs


def build_vocab_from_tokens(seqs: Dict[str, List[str]], min_freq: int = 5) -> Tuple[Dict[str, int], Dict[int, str]]:
    tokens = []
    for s in seqs.values():
        tokens.extend(s)
    vc = pd.Series(tokens).value_counts()
    keep = vc[vc >= min_freq]
    tok2id = {"<PAD>": 0, "<UNK>": 1}
    for t in keep.index.tolist():
        if t not in tok2id:
            tok2id[t] = len(tok2id)
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok


def encode_token_seqs(seqs: Dict[str, List[str]], tok2id: Dict[str, int]) -> Dict[str, List[int]]:
    out = {}
    for k, s in seqs.items():
        out[k] = [tok2id.get(t, tok2id["<UNK>"]) for t in s]
    return out


class NextEventDataset(Dataset):
    def __init__(self, sequences: List[List[int]], context: int = 20):
        self.samples = []
        self.context = context
        for seq in sequences:
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                start = max(0, i - context)
                ctx = seq[start:i]
                tgt = seq[i]
                self.samples.append((ctx, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_next_event(batch, pad_id: int = 0):
    ctxs, tgts = zip(*batch)
    maxlen = max(len(c) for c in ctxs)
    X = np.full((len(ctxs), maxlen), pad_id, dtype=np.int64)
    for i, c in enumerate(ctxs):
        X[i, -len(c) :] = np.array(c, dtype=np.int64)
    y = np.array(tgts, dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


class SimpleTransformerNext(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.emb(x)
        h = self.enc(h)
        last = h[:, -1, :]
        logits = self.fc(last)
        return logits


def train_lanl_predictive(
    auth_gz: str,
    redteam_gz: str,
    out_dir: str,
    window_seconds: int = 7200,
    tol_seconds: int = 0,
    max_auth_lines: int = 2_000_000,
    seed: int = 42,
) -> None:
    ensure_dir(out_dir)
    set_seed(seed)

    print_header("LANL: peek + infer delimiter")
    delim = infer_delim_from_peek(auth_gz)
    print(f"Inferred delimiter: {repr(delim)}")

    rt_delim = infer_delim_from_peek(redteam_gz)
    if rt_delim != delim:
        print(f"Redteam delimiter differs: {repr(rt_delim)}")

    tmin_rt = get_earliest_time_in_gz(redteam_gz, delim=rt_delim)
    t0, t1 = tmin_rt, tmin_rt + window_seconds
    print(f"Earliest redteam time: {tmin_rt}")
    print(f"Extracting window: [{t0}, {t1}] ({window_seconds}s)")

    auth_out = f"{out_dir}/auth_window.csv"
    rt_out = f"{out_dir}/redteam_window.csv"

    na = extract_lanl_window(auth_gz, auth_out, t0, t1, delim=delim, max_lines=max_auth_lines)
    nr = extract_lanl_window(redteam_gz, rt_out, t0, t1, delim=rt_delim, max_lines=None)
    print(f"Wrote auth window lines: {na:,}")
    print(f"Wrote redteam window lines: {nr:,}")

    df_auth = pd.read_csv(auth_out, header=None, names=LANL_AUTH_COLS)
    df_rt = pd.read_csv(rt_out, header=None, names=LANL_RT_COLS)

    df_lab = label_auth_by_redteam(df_auth, df_rt, tol=tol_seconds)
    print_header("LANL: label counts")
    print(df_lab["is_malicious"].value_counts(dropna=False))
    df_lab.to_csv(f"{out_dir}/auth_labeled_window.csv", index=False)

    seqs = build_user_dest_sequences(df_lab)

    tok2id, _ = build_vocab_from_tokens(seqs, min_freq=5)
    seqs_enc = encode_token_seqs(seqs, tok2id)

    sequences = [s for s in seqs_enc.values() if len(s) >= 3]
    if not sequences:
        raise ValueError("Not enough LANL sequences to train the predictive head.")

    tr, va = train_test_split(sequences, test_size=0.2, random_state=seed)

    ds_tr = NextEventDataset(tr, context=20)
    ds_va = NextEventDataset(va, context=20)
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, collate_fn=lambda b: collate_next_event(b, 0))
    dl_va = DataLoader(ds_va, batch_size=512, shuffle=False, collate_fn=lambda b: collate_next_event(b, 0))

    model = SimpleTransformerNext(vocab_size=len(tok2id), d_model=128, nhead=4, num_layers=2).to(device())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    print_header("LANL Predictive Head: training transformer (next dst_comp per user)")
    best_va = float("inf")
    for epoch in range(5):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tqdm(dl_tr, desc=f"epoch {epoch} train", leave=False):
            Xb, yb = Xb.to(device()), yb.to(device())
            opt.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(dl_tr))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in tqdm(dl_va, desc=f"epoch {epoch} val", leave=False):
                Xb, yb = Xb.to(device()), yb.to(device())
                logits = model(Xb)
                va_loss += loss_fn(logits, yb).item()
        va_loss /= max(1, len(dl_va))
        print(f"epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
        if va_loss < best_va:
            best_va = va_loss
            torch.save(model.state_dict(), f"{out_dir}/lanl_predictive_transformer.pt")

    print_header("LANL Predictive Head: scoring surprise")
    model.load_state_dict(torch.load(f"{out_dir}/lanl_predictive_transformer.pt", map_location=device()))
    model.eval()

    df_lab_sorted = df_lab.sort_values(["user", "time"]).copy()
    df_lab_sorted["dst_id"] = df_lab_sorted["dst_comp"].astype(str).map(
        lambda x: tok2id.get(x, tok2id["<UNK>"])
    )

    surprises = np.zeros(len(df_lab_sorted), dtype=np.float64)
    idxs = df_lab_sorted.index.to_numpy()

    with torch.no_grad():
        for user, g in tqdm(df_lab_sorted.groupby("user"), desc="users", leave=False):
            dst_ids = g["dst_id"].tolist()
            if len(dst_ids) < 2:
                continue
            g_idx = g.index.to_numpy()
            for i in range(1, len(dst_ids)):
                start = max(0, i - 20)
                ctx = dst_ids[start:i]
                tgt = dst_ids[i]
                Xb = torch.tensor([ctx], dtype=torch.long).to(device())
                logits = model(Xb)
                prob = torch.softmax(logits, dim=-1)[0, tgt].item()
                prob = max(prob, 1e-12)
                surprises[np.where(idxs == g_idx[i])[0][0]] = -math.log(prob)

    df_scored = df_lab_sorted.copy()
    df_scored["surprise"] = surprises

    normal_sur = df_scored[df_scored["is_malicious"] == 0]["surprise"].values
    if len(normal_sur):
        thr = float(np.percentile(normal_sur, 95))
    else:
        thr = float(np.percentile(df_scored["surprise"].values, 95))
    df_scored["pred_malicious"] = (df_scored["surprise"] > thr).astype(int)

    print_header("LANL Predictive Head: report vs redteam labels")
    y_true = df_scored["is_malicious"].values.astype(int)
    y_pred = df_scored["pred_malicious"].values.astype(int)
    print(classification_report(y_true, y_pred, digits=4))
    save_json(
        classification_report(y_true, y_pred, digits=4, output_dict=True),
        f"{out_dir}/lanl_report.json",
    )
    save_json({"threshold_surprise": thr, "vocab_size": len(tok2id)}, f"{out_dir}/lanl_meta.json")
    df_scored.to_csv(f"{out_dir}/lanl_scored.csv", index=False)
