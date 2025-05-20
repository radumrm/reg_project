import pandas as pd

def prepare_dataset():
    df = pd.read_csv("vienna_weekdays.csv")
    to_remove = ['room_shared', 'multi', 'biz', 'lng', 'lat', 'attr_index', 'rest_index']
    df = df[[col for col in df.columns if col not in to_remove]]
    df.to_csv("dataset.csv", index=False)

prepare_dataset()