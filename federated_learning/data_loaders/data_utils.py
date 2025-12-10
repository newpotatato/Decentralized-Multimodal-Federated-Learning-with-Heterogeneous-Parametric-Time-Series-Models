import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union


def load_mcc_series(base_path: Path) -> pd.DataFrame:
    """Load MCC transactions; keeps numeric category columns and date."""
    # First try local v2/data directory, then fall back to repo root
    v2_mcc_path = Path(__file__).parent / "data" / "dat_mcc.csv"
    mcc_path = v2_mcc_path if v2_mcc_path.exists() else base_path / "01_data_transactions" / "dat_mcc.csv"
    if not mcc_path.exists():
        raise FileNotFoundError(f"Missing {mcc_path}; place real MCC data there.")

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


def build_clients_from_mcc(
    mcc_df: pd.DataFrame,
    exogenous: Optional[Union[pd.Series, pd.DataFrame]],
    n_clients: int,
    min_points: int = 150,
) -> List[pd.DataFrame]:
    """Split MCC categories across clients and attach exogenous factors (Series or DataFrame)."""
    num_cols = [c for c in mcc_df.columns if c != "date" and pd.api.types.is_numeric_dtype(mcc_df[c])]
    chunks = _chunk_columns(num_cols, n_clients)

    exog_frame: Optional[pd.DataFrame] = None
    if exogenous is not None:
        exog_frame = _align_exog_frame(exogenous, len(mcc_df))

    clients: List[pd.DataFrame] = []
    for cols in chunks:
        client = pd.DataFrame()
        client["date"] = mcc_df["date"].copy()
        client["amt"] = mcc_df[cols].sum(axis=1)
        if exog_frame is not None:
            for col in exog_frame.columns:
                client[col] = exog_frame[col].values
        client = client.dropna(subset=["amt"])
        if len(client) >= min_points:
            clients.append(client.reset_index(drop=True))
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
