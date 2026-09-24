import os
import json
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from scipy import stats

def augment_dataset(target_total=10000):
    seed_path = 'data/raw/producthunt_scraped_2000.csv'
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed dataset not found at {seed_path}")
        
    df_real = pd.read_csv(seed_path)
    real_count = len(df_real)
    print(f"Loaded Real Seed Dataset: {real_count:,} observations from {seed_path}")
    
    # Harmonize scraped column names if needed
    if 'upvotes' in df_real.columns and 'votesCount' not in df_real.columns:
        df_real['votesCount'] = df_real['upvotes']
    if 'comments' in df_real.columns and 'commentsCount' not in df_real.columns:
        df_real['commentsCount'] = df_real['comments']
    if 'categories' in df_real.columns and 'primary_topic' not in df_real.columns:
        df_real['primary_topic'] = df_real['categories'].fillna('Other')
    if 'team_size' in df_real.columns and 'maker_count' not in df_real.columns:
        df_real['maker_count'] = df_real['team_size']
        
    # Ensure default follower columns exist
    if 'maker_total_followers' not in df_real.columns:
        df_real['maker_total_followers'] = 100.0
    if 'hunter_followers_count' not in df_real.columns:
        df_real['hunter_followers_count'] = 500.0
    if 'topic_count' not in df_real.columns:
        df_real['topic_count'] = 3
    if 'weekday' not in df_real.columns:
        df_real['weekday'] = 2
    if 'month' not in df_real.columns:
        df_real['month'] = 6
    if 'year' not in df_real.columns:
        df_real['year'] = 2024
    if 'has_video' not in df_real.columns:
        df_real['has_video'] = False
    if 'hunter_prev_launches' not in df_real.columns:
        df_real['hunter_prev_launches'] = 0
    if 'hunter_prev_avg_votes' not in df_real.columns:
        df_real['hunter_prev_avg_votes'] = 0.0
    if 'maker_prev_launches' not in df_real.columns:
        df_real['maker_prev_launches'] = 0
    if 'maker_prev_avg_votes' not in df_real.columns:
        df_real['maker_prev_avg_votes'] = 0.0

    # Identify how many synthetic rows to generate to reach ~10,000 total
    synth_needed = max(1000, target_total - real_count)
    print(f"Target Total Dataset Size: {target_total:,}")
    print(f"Generating {synth_needed:,} synthetic rows via Gretel AI / CTGAN generative augmentation...")
    
    # Prepare features for synthesis
    # We synthesize the core modeling features to maintain high statistical fidelity
    synth_columns = [
        'votesCount', 'commentsCount', 'launch_hour', 'weekday', 'month', 'year',
        'is_featured', 'description_length', 'tagline_length', 'topic_count',
        'maker_count', 'maker_total_followers', 'hunter_followers_count',
        'media_count', 'has_video', 'self_launched', 'primary_topic',
        'hunter_prev_launches', 'hunter_prev_avg_votes',
        'maker_prev_launches', 'maker_prev_avg_votes'
    ]
    
    df_train = df_real[synth_columns].copy()
    
    # Handle top topics for synthesizer stability
    top_25_topics = df_train['primary_topic'].value_counts().head(25).index.tolist()
    df_train['primary_topic'] = df_train['primary_topic'].apply(lambda x: x if x in top_25_topics else 'Other')
    
    # Auto-detect metadata
    metadata = Metadata.detect_from_dataframe(df_train)
    
    print("\nTraining Generative Tabular Synthesizer (epochs=300)...")
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=300,
        batch_size=200,
        verbose=False
    )
    
    synthesizer.fit(df_train)
    print("Training complete. Sampling synthetic records...")
    
    synthetic_df = synthesizer.sample(num_rows=synth_needed)
    
    # Post-process synthetic data to ensure valid ranges
    synthetic_df['votesCount'] = np.maximum(0, synthetic_df['votesCount'].round(0))
    synthetic_df['commentsCount'] = np.maximum(0, synthetic_df['commentsCount'].round(0))
    synthetic_df['launch_hour'] = np.clip(synthetic_df['launch_hour'].round(0), 0, 23).astype(int)
    synthetic_df['weekday'] = np.clip(synthetic_df['weekday'].round(0), 0, 6).astype(int)
    synthetic_df['month'] = np.clip(synthetic_df['month'].round(0), 1, 12).astype(int)
    synthetic_df['year'] = np.clip(synthetic_df['year'].round(0), 2023, 2026).astype(int)
    synthetic_df['description_length'] = np.maximum(0, synthetic_df['description_length'].round(0)).astype(int)
    synthetic_df['tagline_length'] = np.maximum(4, synthetic_df['tagline_length'].round(0)).astype(int)
    synthetic_df['media_count'] = np.maximum(0, synthetic_df['media_count'].round(0)).astype(int)
    synthetic_df['maker_count'] = np.maximum(0, synthetic_df['maker_count'].round(0)).astype(int)
    synthetic_df['topic_count'] = np.clip(synthetic_df['topic_count'].round(0), 1, 10).astype(int)
    synthetic_df['maker_total_followers'] = np.maximum(0, synthetic_df['maker_total_followers'].round(0))
    synthetic_df['hunter_followers_count'] = np.maximum(0, synthetic_df['hunter_followers_count'].round(0))
    synthetic_df['hunter_prev_launches'] = np.maximum(0, synthetic_df['hunter_prev_launches'].round(0)).astype(int)
    synthetic_df['hunter_prev_avg_votes'] = np.maximum(0, synthetic_df['hunter_prev_avg_votes'].round(1))
    synthetic_df['maker_prev_launches'] = np.maximum(0, synthetic_df['maker_prev_launches'].round(0)).astype(int)
    synthetic_df['maker_prev_avg_votes'] = np.maximum(0, synthetic_df['maker_prev_avg_votes'].round(1))
    synthetic_df['is_featured'] = synthetic_df['is_featured'].astype(bool)
    synthetic_df['has_video'] = synthetic_df['has_video'].astype(bool)
    synthetic_df['self_launched'] = synthetic_df['self_launched'].astype(bool)
    
    # Label provenance
    df_real_subset = df_train.copy()
    df_real_subset['is_synthetic'] = False
    
    synthetic_df['is_synthetic'] = True
    
    # Combined final dataset
    df_combined = pd.concat([df_real_subset, synthetic_df], ignore_index=True)
    # Shuffle
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    out_v2_path = 'data/raw/synthetic_producthunt_10000_v2.csv'
    df_combined.to_csv(out_v2_path, index=False)
    
    print(f"\n==========================================")
    print(f"Final Analytical Dataset Saved: {out_v2_path}")
    print(f"Total Rows: {len(df_combined):,}")
    print(f"Real Seed Rows: {len(df_real_subset):,} ({len(df_real_subset)/len(df_combined)*100:.1f}%)")
    print(f"Gretel Synthetic Rows: {len(synthetic_df):,} ({len(synthetic_df)/len(df_combined)*100:.1f}%)")
    print(f"==========================================")
    
    # 4. Fidelity Audit: Real vs Synthetic
    print("\n=== REAL VS SYNTHETIC FIDELITY AUDIT ===")
    fidelity_report = []
    
    numeric_check = [
        'votesCount', 'commentsCount', 'launch_hour', 'weekday', 'description_length',
        'tagline_length', 'media_count', 'maker_count', 'hunter_followers_count', 'maker_total_followers'
    ]
    
    for col in numeric_check:
        r_mean = df_real_subset[col].mean()
        s_mean = synthetic_df[col].mean()
        r_med = df_real_subset[col].median()
        s_med = synthetic_df[col].median()
        r_std = df_real_subset[col].std()
        s_std = synthetic_df[col].std()
        
        # Kolmogorov-Smirnov 2-sample test
        ks_stat, ks_p = stats.ks_2samp(df_real_subset[col], synthetic_df[col])
        
        fidelity_report.append({
            'Feature': col,
            'Real_Mean': round(r_mean, 2),
            'Synth_Mean': round(s_mean, 2),
            'Real_Med': round(r_med, 2),
            'Synth_Med': round(s_med, 2),
            'Real_Std': round(r_std, 2),
            'Synth_Std': round(s_std, 2),
            'KS_Stat': round(ks_stat, 4),
            'KS_pValue': f"{ks_p:.2e}"
        })
        
    fidelity_df = pd.DataFrame(fidelity_report)
    print(fidelity_df.to_string(index=False))
    
    fidelity_out_path = 'data/processed/fidelity_audit.json'
    os.makedirs('data/processed', exist_ok=True)
    with open(fidelity_out_path, 'w') as f:
        json.dump(fidelity_report, f, indent=2)
        
    return df_combined

if __name__ == '__main__':
    augment_dataset(target_total=10000)
