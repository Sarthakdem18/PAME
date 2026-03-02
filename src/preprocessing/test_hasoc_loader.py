from load_aux_data import load_hasoc

df_hate = load_hasoc("../../data/aux_hate/hasoc_train.tsv")

print(df_hate.head())
print("\nHate label distribution:")
print(df_hate["hate"].value_counts())
