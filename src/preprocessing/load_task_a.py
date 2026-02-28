import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "aux_hate" / "task_a"

def load_task_a_train():
    df = pd.read_excel(DATA_DIR / "Train_Task_A.xlsx")
    return _normalize(df)

def load_task_a_val():
    df = pd.read_excel(DATA_DIR / "Val_Task_A.xlsx")
    return _normalize(df)

def _normalize(df):
    df = df.rename(columns={
        "Tweet": "text",
        "Hate": "hate",
        "Fake": "fake"
    })

    df["hate"] = df["hate"].astype(int)
    df["fake"] = df["fake"].astype(int)

    df["faux"] = ((df["hate"] == 1) & (df["fake"] == 1)).astype(int)

    return df[["text", "hate", "fake", "faux"]]
