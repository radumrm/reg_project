import pandas as pd
import numpy as np

def prepare_dataset():
    df = pd.read_csv("vienna_weekdays.csv")

    # Scoatem coloanele care nu sunt relevante
    to_remove = ['room_shared', 'multi', 'biz', 'attr_index', 'rest_index']
    df = df[[col for col in df.columns if col not in to_remove]]

    # Introdugem zgomot in realSum
    noise = np.random.normal(loc = 0, scale = 0.05, size = len(df))
    df['realSum'] = df['realSum'] * (noise + 1)

    # Introducem valori lipsa la realSum, dist si guest_satisfaction de 5%
    for column in ['realSum', 'guest_satisfaction_overall', 'realSum']:
        to_Remove = np.random.choice(df.index, int(0.05 * len(df)), replace = False)
        df.loc[to_Remove, column] = np.nan
    
    df.to_csv("data_set.csv", index=False)

prepare_dataset()