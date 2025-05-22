import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, confusion_matrix
from aberant_removal import aberante_removal
from sklearn.model_selection import train_test_split
from prepare_dataset import prepare_dataset
from analyze_data import analyze_data

df_train, df_test = prepare_dataset()
df_train, df_test = analyze_data(df_train, df_test)

# Concatenam seturile de date
df = pd.concat([df_train, df_test])

# Din cauza valoriilor foarte imprastiate folosim o transformare log
df['realSum'] = np.log1p(df['realSum'])

print(df.describe())

# One hot encoding pentru room-type
df = pd.get_dummies(df, columns=['room_type'], drop_first=True)

# Retinem coloanele numerice
aux = df.select_dtypes(include='number').columns
cols = [col for col in aux if col not in ['Unnamed: 0']]
df = aberante_removal(df, cols)

# Normalizam variabilele numerice
scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

# Standardizam variabilele numerice
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])

# Impartim setul de date 
X = df.drop(columns=['Unnamed: 0', 'realSum'])
y = df['realSum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model and evaluate
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2: {r2:.2f}")

# Vizualizăm matricea de confuzie
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.grid(True)
plt.title("Matricea de erori")
plt.xlabel("Realitate")
plt.ylabel("Predictii")
plt.savefig("ERORI.png")
plt.close()

