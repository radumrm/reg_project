import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def analyze_data():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    # Verificam daca s-au incarcat corect subseturile de data
    print(df_train.head())
    print(df_test.head())

    # Afisam tipurile de date ale dataset-ului
    print(df_train.dtypes)

    # Analizam datatele lipsa
    print(df_train.isnull().sum())

    # Observam ca intmapinam campuri lipsa pentru 'realSum'
    # 'guest_satifaction' si 'dist'

    # Vom scoate randurile care au realSum din subseturi deoarece acestea ne modifica acuratetea
    df_train = df_train[~df_train['realSum'].isnull()]
    df_test = df_test[~df_test['realSum'].isnull()]

    # Vom completa in campurile lipsa din coloanele 'guest_satisfaction_overall' si 'dist' media
    guest_mean = df_train['guest_satisfaction_overall'].mean()
    dist_mean = df_train['dist'].mean()
    print(f"Media guest_satisfaction_overall: {guest_mean}")
    print(f"Media dist: {dist_mean}\n")

    df_train.loc[:, 'guest_satisfaction_overall'] = df_train['guest_satisfaction_overall'].fillna(guest_mean)
    df_test.loc[:, 'guest_satisfaction_overall'] = df_test['guest_satisfaction_overall'].fillna(dist_mean)

    df_train.loc[:, 'dist'] = df_train['dist'].fillna(dist_mean)
    df_test.loc[:, 'dist'] = df_test['dist'].fillna(dist_mean)

    # Stat istici descriptive
    print(df_train.describe())

    # Histograma pentru valorile numerice
    aux = df_train.select_dtypes(include='number').columns
    cols = [col for col in aux if col not in ['lng', 'lat', 'Unnamed: 0']]

    plt.figure(figsize=(16, 10))
    for i in range(8):
        plt.subplot(2, 4, i + 1)

        if cols[i] == 'realSum':
            data = df_train[df_train[cols[i]] < 3000][cols[i]]
            sns.histplot(data, bins = 20)
        else:
            sns.histplot(df_train[cols[i]], bins = 10)
        plt.title(f"Histograma pentru {cols[i]}")
        plt.xlabel(cols[i])
        plt.ylabel("Frecventa")

    plt.tight_layout(pad=3)
    plt.savefig("HISTOGRAM.png")
    plt.close()

analyze_data()