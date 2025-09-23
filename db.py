# db.py
import re
import sqlite3
import unicodedata
import pandas as pd
import threading
from typing import Dict, List

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _slugify(name: str) -> str:
    s = str(name)
    s = _strip_accents(s).lower().strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"c_{s}"
    if s in {"index"}:
        s = "idx"
    return s

def _make_unique(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")  # nivel, nivel_1, nivel_2...
    return out

class CSVDb:
    def __init__(self):
        # Permite usar la MISMA conexión en varios hilos de Flask
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.colmap: Dict[str, str] = {}
        self.last_df: pd.DataFrame | None = None

    def load_dataframe(self, df: pd.DataFrame, table_name: str):
        df = df.copy()
        raw_cols = [str(c) for c in df.columns]
        safe_cols = [_slugify(c) for c in raw_cols]
        uniq_cols = _make_unique(safe_cols)
        self.colmap = {orig: final for orig, final in zip(raw_cols, uniq_cols)}
        df.columns = uniq_cols
        self.last_df = df
        with self._lock:
            df.to_sql(table_name, self.conn, if_exists="replace", index=False)
            # A partir de aquí, solo lectura
            self.conn.execute("PRAGMA query_only = 1")

    def describe_schema(self, table_name: str) -> str:
        if self.last_df is None:
            raise RuntimeError("No hay DataFrame cargado.")
        df = self.last_df
        lines = [f"TABLE {table_name} ("]
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            nulls = int(s.isna().sum())
            nunique = int(s.nunique(dropna=True))
            samples = s.dropna().astype(str).unique()[:6]
            example = ", ".join(map(str, samples))
            lines.append(f"  {col}  -- dtype={dtype}, nulls={nulls}, unique={nunique}, examples=[{example}]")
        lines.append(")")
        lines.append(f"-- rows={len(df)}")
        changed = [k for k, v in self.colmap.items() if k != v]
        if changed:
            lines.append("-- column name mapping (original -> final):")
            for k, v in self.colmap.items():
                if k != v:
                    lines.append(f"--   {k} -> {v}")
        return "\n".join(lines)

    def query(self, sql: str) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self.conn)
