import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Importuri pentru heatmap pe harta
import plotly.graph_objects as go

def analyze_data():
    df_train = pd.read_csv("test.csv")
    df_test = pd.read_csv("train.csv")

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
    print(f"\nMedia guest_satisfaction_overall: {guest_mean}")
    print(f"Media dist: {dist_mean}\n")

    df_train.loc[:, 'guest_satisfaction_overall'] = df_train['guest_satisfaction_overall'].fillna(guest_mean)
    df_test.loc[:, 'guest_satisfaction_overall'] = df_test['guest_satisfaction_overall'].fillna(dist_mean)

    df_train.loc[:, 'dist'] = df_train['dist'].fillna(dist_mean)
    df_test.loc[:, 'dist'] = df_test['dist'].fillna(dist_mean)

    # Statistici descriptive
    print(df_test.describe())

    # Histograma pentru valorile numerice
    aux = df_train.select_dtypes(include='number').columns
    cols = [col for col in aux if col not in ['lng', 'lat', 'Unnamed: 0']]

    plt.figure(figsize=(17, 10))
    for i in range(8):
        plt.subplot(2, 4, i + 1)

        if cols[i] == 'realSum':
            # Necesara pentru afisarea corespunzatoare a datelor, ignorand valorile abreante
            data = df_train[df_train[cols[i]] < 3000][cols[i]]
            sns.histplot(data, bins = 20)
        else:
            sns.histplot(df_train[cols[i]], bins = 20)
        plt.title(f"Histograma pentru {cols[i]}")
        plt.xlabel(cols[i])
        plt.ylabel("Frecventa")

    plt.tight_layout(pad = 3)
    plt.savefig("HISTOGRAM.png")
    plt.close()

    # Grafice tip countplot pentru variabile categorice
    cols = ['room_type', 'host_is_superhost']
    plt.figure(figsize=(10, 5))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        sns.countplot(x=cols[i], data=df_train)
        plt.title(f"Countplot pentru {cols[i]}")
        plt.xlabel(f"{cols[i]}")
        plt.ylabel(f"Frecventa")

    plt.tight_layout(pad = 3)
    plt.savefig("COUNTPLOT.png")
    plt.close()

    # Vizualizarea valorilor aberante pentru realSum, dist si metro_dist
    cols = ['realSum', 'dist', 'metro_dist']
    plt.figure(figsize=(17, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x=df_train[cols[i]])
        plt.title(f"Boxplot pentru {cols[i]}")
        plt.xlabel(f"{cols[i]}")

    plt.tight_layout(pad = 3)
    plt.savefig("BOXPLOT.png")
    plt.close()

    # Heatmap pentru valori numerice
    aux = df_train.select_dtypes(include='number').columns
    cols = [col for col in aux if col not in ['lng', 'lat', 'Unnamed: 0']]
    corrm = df_train[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corrm, annot = True, fmt = ".2f", cmap = "Blues")
    plt.title("Heatmap")
    plt.tight_layout()
    plt.savefig("HEATMAP.png")
    plt.close()

    # Analiza relatiilor cu variabila tinta, realSum
    cols = ['dist', 'person_capacity', 'bedrooms']
    # Salvam in data realSum fara valorile aberante
    data = df_train[df_train['realSum'] < 3000]['realSum']
    plt.figure(figsize=(17, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.scatterplot(x = df_train[cols[i]], y = data)
        plt.title(f"Relatia dintre pret si {cols[i]}")
        plt.xlabel(f"{cols[i]}")
        plt.ylabel("Pret")

    plt.tight_layout(pad = 3)
    plt.savefig("SCATTERPLOT.png")
    plt.close()

    #EXTRA heatmap pe harta, sursa: https://www.kaggle.com/code/thedevastator/airbnb-prices-tripadvisor-ratings-starter + FINETUNING
    trace = go.Densitymap(
        lat=df_train["lat"],
        lon=df_train["lng"],
        z=df_train['realSum'],
        radius=100,
        colorscale="Hot",
        opacity=0.9,
        showscale=False,
        colorbar=dict(title=dict(text="Airbnb Prices", side="top"), thickness=20, ticksuffix="€")
    )

    mapbox_style = "carto-positron"
    center_lat = df_train["lat"].mean()
    center_lon = df_train["lng"].mean()
    layout = go.Layout(
    title=dict(
        text="<b>Vienna</b> (Weekdays)",
        font=dict(size=16)
    ),
    mapbox=dict(
        style=mapbox_style,
        center=dict(lat=center_lat, lon=center_lon),
        zoom=10
    ),
    hovermode="closest",
    margin=dict(l=30, r=30, t=50, b=30)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_html("vienna_heatmap.html")
    
    return df_train, df_test