import re
import pandas as pd

def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    df = pd.read_csv("../../data/faux_hate.csv")

    df["clean_text"] = df["text"].apply(clean_text)

    print(df[["text", "clean_text"]].head(10))
