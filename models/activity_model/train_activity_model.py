import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# === 1. Load the CSV ===
csv_path = '../data/csv/full_fake_depression_activity_log.csv'
df = pd.read_csv(csv_path)

# === 2. Preprocess Columns ===
df['Journal_Activity'] = df['Journal_Activity'].map({'Yes': 1, 'No': 0})

# === 3. Prepare Features & Target ===
X = df.drop(columns=['User_ID', 'Date', 'Depression_Score'])
y = df['Depression_Score']

# === 4. Normalize Using MinMaxScaler ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 5. Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 6. Build Keras Copycat Model ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(9,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output between 0–1
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === 7. Train the Model ===
es = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=16, callbacks=[es])

# === 8. Evaluate Accuracy ===
y_pred = model.predict(X_test).flatten()
print("✅ R2 Score:", r2_score(y_test, y_pred))

# === 9. Save Model & Scaler ===
os.makedirs('../activity_model', exist_ok=True)
model.save('../activity_model/activity_model.h5')
joblib.dump(scaler, '../activity_model/scaler.pkl')  # keep this to use for preprocessing

print("✅ activity_model.h5 saved successfully!")
  