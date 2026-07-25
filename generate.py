import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

print("Loading dataset...")

df = pd.read_csv("producthunt_ml_ready.csv")

columns = [
    "rank",
    "votesCount",
    "commentsCount",
    "launch_hour",
    "weekday",
    "month",
    "is_featured",
    "description_length",
    "tagline_length",
    "topic_count",
    "maker_count",
    "maker_total_followers",
    "hunter_followers_count",
    "media_count",
    "has_video",
    "self_launched",
    "primary_topic"
]

missing = [c for c in columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[columns].copy()

# Handle missing values
df["primary_topic"] = df["primary_topic"].fillna("Unknown")

numeric_columns = [
    "rank",
    "votesCount",
    "commentsCount",
    "launch_hour",
    "weekday",
    "month",
    "description_length",
    "tagline_length",
    "topic_count",
    "maker_count",
    "maker_total_followers",
    "hunter_followers_count",
    "media_count"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

boolean_columns = [
    "is_featured",
    "has_video",
    "self_launched"
]

for col in boolean_columns:
    df[col] = df[col].fillna(False).astype(bool)

print("\nTraining Dataset")
print(df.head())
print(df.shape)

print("\nDetecting metadata...")

metadata = Metadata.detect_from_dataframe(df)

print("Training CTGAN...")

synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=1000,
    verbose=True
)

synthesizer.fit(df)

print("Generating synthetic dataset...")

synthetic = synthesizer.sample(num_rows=10000)

synthetic.to_csv(
    "synthetic_producthunt_10000.csv",
    index=False
)

print("\nFinished")
print(synthetic.head())
print(synthetic.shape)
