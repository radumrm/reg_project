import pandas as pd
import numpy as np

def prepare_dataset():
    df = pd.read_csv("vienna_weekdays.csv")

    # Scoatem coloanele care nu sunt relevante
    to_remove = ['room_shared', 'multi', 'biz', 'attr_index', 'rest_index', 'attr_index_norm', 'room_private']
    df = df[[col for col in df.columns if col not in to_remove]]

    # Introdugem zgomot in realSum, guest_satisfaction
    noise = np.random.normal(loc = 0, scale = 0.05, size = len(df))
    df['realSum'] = df['realSum'] * (noise + 1)
    noise = -np.abs(np.random.normal(loc = 0, scale = 0.1, size = len(df)))
    df['guest_satisfaction_overall'] =  df['guest_satisfaction_overall'] * (noise + 1)

    for col in ['realSum', 'guest_satisfaction_overall', 'dist']:
        to_remove = np.random.choice(df.index, int(0.1 * len(df)), replace=False)
        df.loc[to_remove, col] = np.nan
    
    # Schimbam coloana de dormitoare din float64 in int64
    df['bedrooms'] = df['bedrooms'].astype('int64')

    # Amestecam setul de date
    df = df.sample(frac=1).reset_index(drop=True)

    # Impartim dataset-ul in doua subseturi pentru training si testing (80& cu 20%)
    df_train = df.iloc[:round(0.8 * len(df))]
    df_test = df.iloc[round(0.8 * len(df)) + 1:]

    # Exportam cele doua subseturi
    df_train.to_csv('train.csv', index = False)
    df_test.to_csv('test.csv', index = False)
