import pandas as pd

path = "./data/clang8.tsv"
df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
df.columns = ["source", "target"]

df = df[:1000]

df.to_csv("./data/clang8_sample.csv", sep="\t", index=False)
