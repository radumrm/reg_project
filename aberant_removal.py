def aberante_removal(df, cols):
    df_clean = df.copy()
    for col in cols:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        df_clean = df_clean[df_clean[col].between(lower, upper)]
    
    return df_clean.reset_index(drop=True)