import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Information profiles S_i for Jaccard / graph (manuscript): modalities + partition id.

_BUNDLE_DIR = Path(__file__).resolve().parent / "data"
_SYNTHETIC_MCC = _BUNDLE_DIR / "dat_mcc_synthetic_smoke.csv"


def mcc_csv_candidate_paths(base_path: Path) -> List[Path]:
    """
    Ordered search locations for real MCC CSV under the repository root.
    Prefer full export, then standard layout (README), then data_LVP mirror.
    """
    base = Path(base_path).resolve()
    return [
        base / "01_data_transactions" / "dat_mcc_full.csv",
        base / "01_data_transactions" / "dat_mcc.csv",
        base / "data" / "transactions" / "dat_mcc_full.csv",
        base / "data" / "transactions" / "dat_mcc.csv",
        base / "data_LVP" / "01_data_transactions" / "dat_mcc_full.csv",
        base / "data_LVP" / "01_data_transactions" / "dat_mcc.csv",
    ]


def resolve_mcc_csv_path(base_path: Path) -> Path:
    """
    Resolve path to MCC CSV. Real data under base_path is always preferred over
    the bundled tiny CSV. Set FL_ALLOW_SYNTHETIC_MCC=1 to allow the bundle for CI/smoke only.
    """
    for p in mcc_csv_candidate_paths(base_path):
        if p.is_file():
            return p

    allow = os.environ.get("FL_ALLOW_SYNTHETIC_MCC", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if allow and _SYNTHETIC_MCC.is_file():
        warnings.warn(
            "Using bundled synthetic MCC (set FL_ALLOW_SYNTHETIC_MCC=1). "
            "Not for publication runs.",
            stacklevel=2,
        )
        return _SYNTHETIC_MCC

    tried = "\n  ".join(str(p) for p in mcc_csv_candidate_paths(base_path))
    raise FileNotFoundError(
        "No real MCC dataset found. Expected one of:\n  "
        f"{tried}\n"
        "Copy dat_mcc.csv (or dat_mcc_full.csv) into 01_data_transactions/ at the repo root, "
        "or set FL_ALLOW_SYNTHETIC_MCC=1 to use the tiny bundled sample."
    )


def load_mcc_series(base_path: Path) -> pd.DataFrame:
    """Load MCC transactions; keeps numeric category columns and date."""
    mcc_path = resolve_mcc_csv_path(base_path)

    df = pd.read_csv(mcc_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # if no date column, synthesize a simple daily index
        df["date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("dat_mcc.csv has no numeric category columns.")

    df = df[["date"] + num_cols].fillna(0)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_news_exogenous(base_path: Path) -> pd.Series:
    """Load Fontanka news; returns daily sentiment or counts as a Series."""
    candidates = [
        Path(__file__).parent / "data" / "news_clustered_final.csv",
        base_path / "02_data_fontanka" / "news_clustered_final.csv",
        base_path / "02_data_fontanka" / "fontanka_news_result.csv",
    ]

    df_news = None
    used_path: Optional[Path] = None
    for path in candidates:
        if not path.exists():
            continue
        for sep in [";", ","]:
            try:
                df_news = pd.read_csv(path, encoding="cp1251", sep=sep)
                used_path = path
                break
            except Exception:
                df_news = None
        if df_news is not None:
            break

    if df_news is None:
        raise FileNotFoundError("Fontanka news file not found in 02_data_fontanka.")

    if "date" not in df_news.columns and "Date" in df_news.columns:
        df_news = df_news.rename(columns={"Date": "date"})
    df_news["date"] = pd.to_datetime(df_news["date"], errors="coerce")
    df_news = df_news.dropna(subset=["date"])

    if "sentiment_score" in df_news.columns:
        daily = (
            df_news.groupby(df_news["date"].dt.date)["sentiment_score"].mean().sort_index()
        )
    else:
        daily = df_news.groupby(df_news["date"].dt.date).size().sort_index()

    series = pd.Series(daily.values.astype(float), index=pd.to_datetime(daily.index))
    series.name = used_path.name if used_path else "news_exog"
    return series


def load_moex_series(base_path: Path) -> pd.DataFrame:
    """Load MOEX stock data; returns daily price aggregations as DataFrame."""
    # Try different base paths
    candidates = [
        Path(__file__).parent / "data" / "moex_data.csv",
        base_path / "01_data_transactions" / "moex_data.csv",
        base_path.parent / "01_data_transactions" / "moex_data.csv",  # If base_path is a subdirectory
        Path("01_data_transactions") / "moex_data.csv",  # Current working dir
    ]

    df_moex = None
    used_path: Optional[Path] = None
    for path in candidates:
        if path.exists():
            try:
                df_moex = pd.read_csv(path, index_col=0)
                used_path = path
                print(f"  Loaded MOEX from: {used_path}")
                break
            except Exception as e:
                df_moex = None

    if df_moex is None:
        raise FileNotFoundError(f"MOEX data file (moex_data.csv) not found. Tried: {candidates}")

    # Convert index to datetime
    df_moex.index = pd.to_datetime(df_moex.index, errors="coerce")
    df_moex = df_moex.dropna(how="all")
    
    # Add 'date' column for compatibility with other data sources
    df_moex["date"] = df_moex.index
    df_moex = df_moex.reset_index(drop=True)
    
    return df_moex


def _chunk_columns(cols: List[str], n_clients: int) -> List[List[str]]:
    step = max(1, len(cols) // n_clients)
    chunks: List[List[str]] = []
    for i in range(n_clients):
        start = i * step
        end = (i + 1) * step if i < n_clients - 1 else len(cols)
        slice_cols = cols[start:end] or cols[-step:]
        chunks.append(slice_cols)
    return chunks


def _chunk_columns_strided(cols: List[str], n_clients: int) -> List[List[str]]:
    """
    Assign column j to client (j % n_clients). Stronger non-IID than contiguous blocks:
    each client mixes categories from across the full MCC spectrum.
    """
    chunks: List[List[str]] = [[] for _ in range(n_clients)]
    for j, c in enumerate(cols):
        chunks[j % n_clients].append(c)
    return chunks


def _chunk_columns_random(
    cols: List[str],
    n_clients: int,
    *,
    seed: Optional[int] = None,
    strided: bool = False,
) -> List[List[str]]:
    """Shuffle columns before splitting.

    This creates a more irregular client partition than contiguous or strided layouts.
    If strided=True, the shuffled columns are assigned round-robin after shuffling.
    """
    shuffled = list(cols)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    if strided:
        return _chunk_columns_strided(shuffled, n_clients)
    return _chunk_columns(shuffled, n_clients)


def build_clients_from_mcc(
    mcc_df: pd.DataFrame,
    exogenous: Optional[Union[pd.Series, pd.DataFrame]],
    n_clients: int,
    min_points: int = 150,
    column_partition: str = "contiguous",
    partition_seed: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Split MCC categories across clients and attach exogenous factors (Series or DataFrame).

    column_partition:
        contiguous — default, adjacent column blocks per client (milder non-IID).
        strided — round-robin over categories (stronger cross-client heterogeneity).
        random — shuffled contiguous blocks.
        random_strided — shuffled round-robin over categories.
    """
    num_cols = [c for c in mcc_df.columns if c != "date" and pd.api.types.is_numeric_dtype(mcc_df[c])]
    part = (column_partition or "contiguous").strip().lower()
    if part == "strided":
        chunks = _chunk_columns_strided(num_cols, n_clients)
    elif part == "contiguous":
        chunks = _chunk_columns(num_cols, n_clients)
    elif part == "random":
        chunks = _chunk_columns_random(num_cols, n_clients, seed=partition_seed, strided=False)
    elif part == "random_strided":
        chunks = _chunk_columns_random(num_cols, n_clients, seed=partition_seed, strided=True)
    else:
        raise ValueError(
            f"column_partition must be 'contiguous', 'strided', 'random', or 'random_strided', got {column_partition!r}"
        )

    exog_frame: Optional[pd.DataFrame] = None
    if exogenous is not None:
        exog_frame = _align_exog_frame(exogenous, len(mcc_df))

    clients: List[pd.DataFrame] = []
    for idx, cols in enumerate(chunks):
        client = pd.DataFrame()
        client["date"] = mcc_df["date"].copy()
        client["amt"] = mcc_df[cols].sum(axis=1)
        if exog_frame is not None:
            for col in exog_frame.columns:
                client[col] = exog_frame[col].values
        client = client.dropna(subset=["amt"])
        if len(client) >= min_points:
            client = client.reset_index(drop=True)
            # Keep source-column metadata for information profile construction.
            client.attrs["source_columns"] = list(cols)
            client.attrs["partition_mode"] = part
            client.attrs["client_index"] = idx
            clients.append(client)
    return clients


def build_clients_from_moex(
    moex_df: pd.DataFrame,
    n_clients: int,
    min_points: int = 150,
) -> List[pd.DataFrame]:
    """Split MOEX tickers across clients, each client gets individual ticker prices."""
    # Get all ticker columns (numeric columns excluding 'date')
    ticker_cols = [c for c in moex_df.columns if c != "date" and pd.api.types.is_numeric_dtype(moex_df[c])]
    chunks = _chunk_columns(ticker_cols, n_clients)

    clients: List[pd.DataFrame] = []
    for cols in chunks:
        client = pd.DataFrame()
        client["date"] = moex_df["date"].copy()
        # For MOEX, aggregate ticker prices (mean of selected tickers for this client)
        client["amt"] = moex_df[cols].mean(axis=1)
        client = client.dropna(subset=["amt"])
        if len(client) >= min_points:
            clients.append(client.reset_index(drop=True))
    return clients


def _align_exog_frame(exog: Union[pd.Series, pd.DataFrame], target_len: int) -> pd.DataFrame:
    if isinstance(exog, pd.Series):
        exog = exog.to_frame(name="exog")
    exog = exog.sort_index()
    out = pd.DataFrame(index=range(target_len))
    for col in exog.columns:
        values = exog[col].values.astype(float)
        if len(values) >= target_len:
            aligned = values[:target_len]
        else:
            padded = np.zeros(target_len, dtype=float)
            padded[: len(values)] = values
            if len(values) > 0:
                padded[len(values) :] = values[-1]
            aligned = padded
        out[col] = aligned
    return out


def train_test_split_series(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = max(1, int(len(df) * (1 - test_ratio)))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def build_client_information_profiles(
    clients: List[pd.DataFrame], data_source: str
) -> List[frozenset]:
    """
    Build discrete information profiles S_i (multimodal FL manuscript).

    Previous implementation used unique partition tags only, which made Jaccard almost
    constant across all client pairs. Here we encode comparable statistical buckets from
    each client's local series and modality tags, yielding a more informative similarity graph.
    """
    if not clients:
        return []

    stats: List[dict] = []
    for df in clients:
        amt = np.asarray(df["amt"].values, dtype=float)
        amt = np.nan_to_num(amt, nan=0.0, posinf=0.0, neginf=0.0)
        if amt.size == 0:
            stats.append({"mean": 0.0, "std": 0.0, "nz": 0.0, "trend": 0.0, "ncols": 0.0})
            continue

        x = np.arange(amt.size, dtype=float)
        trend = 0.0
        if amt.size >= 3:
            try:
                trend = float(np.polyfit(x, amt, 1)[0])
            except Exception:
                trend = 0.0

        src_cols = df.attrs.get("source_columns", []) if hasattr(df, "attrs") else []
        stats.append(
            {
                "mean": float(np.mean(amt)),
                "std": float(np.std(amt)),
                "nz": float(np.mean(np.abs(amt) > 1e-12)),
                "trend": trend,
                "ncols": float(len(src_cols)),
            }
        )

    def _edges(values: List[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        q = np.quantile(arr, [0.25, 0.5, 0.75])
        return np.asarray(q, dtype=float)

    edges_mean = _edges([s["mean"] for s in stats])
    edges_std = _edges([s["std"] for s in stats])
    edges_nz = _edges([s["nz"] for s in stats])
    edges_trend = _edges([s["trend"] for s in stats])
    edges_ncols = _edges([s["ncols"] for s in stats])

    def _bin(v: float, e: np.ndarray) -> int:
        return int(np.searchsorted(e, float(v), side="right"))

    profiles: List[frozenset] = []
    for idx, df in enumerate(clients):
        tags = {"base", "target", f"source:{data_source}"}
        if data_source == "mcc":
            tags.add("domain:mcc")
        elif data_source == "moex":
            tags.add("domain:moex")
        elif data_source == "news":
            tags.add("domain:news")
        else:
            tags.add("domain:generic")

        s = stats[idx]
        tags.add(f"stat:mean:q{_bin(s['mean'], edges_mean)}")
        tags.add(f"stat:std:q{_bin(s['std'], edges_std)}")
        tags.add(f"stat:nz:q{_bin(s['nz'], edges_nz)}")
        tags.add(f"stat:trend:q{_bin(s['trend'], edges_trend)}")
        tags.add(f"meta:ncols:q{_bin(s['ncols'], edges_ncols)}")

        src_cols = df.attrs.get("source_columns", []) if hasattr(df, "attrs") else []
        id_buckets = set()
        for c in src_cols:
            text = str(c)
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                id_buckets.add(int(digits) // 1000)
        for b in sorted(id_buckets):
            tags.add(f"mcc_bucket:{b}")

        for c in df.columns:
            if str(c).startswith("exog_"):
                tags.add(f"modality:{c}")
        profiles.append(frozenset(tags))
    return profiles
