import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_sample_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "src_ip": [f"10.0.0.{i % 10}" for i in range(n)],
            "dst_ip": [f"10.0.1.{i % 15}" for i in range(n)],
            "src_port": rng.randint(1, 65535, n),
            "dst_port": rng.randint(1, 65535, n),
            "http_uri": rng.choice(["/a", "/b", "-"], n),
            "http_user_agent": rng.choice(["ua1", "ua2", "-"], n),
            "dns_query": rng.choice(["example.com", "-"], n),
            "ssl_subject": rng.choice(["CN=test", "-"], n),
            "ssl_issuer": rng.choice(["CA", "-"], n),
            "http_orig_mime_types": rng.choice(["text/html", "-"], n),
            "http_resp_mime_types": rng.choice(["text/plain", "-"], n),
            "bytes": rng.exponential(1000, n),
            "packets": rng.poisson(10, n),
            "duration": rng.exponential(1.0, n),
            "timestamp": np.arange(n),
        }
    )
    df["label"] = (df["bytes"] > np.median(df["bytes"])).astype(int)
    df["type"] = np.where(df["label"] == 1, "attack", "normal")
    return df


def make_graph_df(n: int = 200, seed: int = 11) -> pd.DataFrame:
    df = make_sample_df(n=n, seed=seed)
    df["src_ip"] = [f"192.168.0.{i % 8}" for i in range(n)]
    df["dst_ip"] = [f"192.168.1.{i % 12}" for i in range(n)]
    return df
