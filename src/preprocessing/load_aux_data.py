import pandas as pd

def load_hasoc(path: str):
    df = pd.read_csv(path, sep="\t")

    df = df[["text", "label"]]
    # HASOC mapping
    df["hate"] = df["label"].apply(
        lambda x: 1 if str(x).strip() == "HOF" else 0
    )

    return df[["text", "hate"]]

def load_fake_aux(path: str):
    df = pd.read_csv(path)

    df = df[["text", "label"]]
    # Fake-news mapping

    df["fake"] = df["label"].apply(
        lambda x: 1 if str(x).strip() == "FAKE" else 0
    )
    return df[["text", "fake"]]

def load_faux_hate(path: str):
    df = pd.read_csv(path)

    df = df[["text", "fake", "hate"]]
    df["fake"] = df["fake"].astype(int)
    df["hate"] = df["hate"].astype(int)

    return df
