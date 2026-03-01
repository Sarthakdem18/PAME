from load_aux_data import load_fake_aux

df_fake = load_fake_aux("../../data/aux_fake/fake_news_aux.csv")

print(df_fake.head())
print("\nFake label distribution:")
print(df_fake["fake"].value_counts())
