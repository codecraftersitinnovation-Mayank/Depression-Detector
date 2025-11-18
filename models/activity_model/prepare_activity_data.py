import pandas as pd
import os

# Use your preferred relative path
CSV_PATH = '../data/csv/activity_logs.csv'
OUTPUT_PATH = '../data/csv/activity_labeled.csv'

# Load dataset
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = df.columns.str.strip()

# Apply labeling rules
labels = []
for index, row in df.iterrows():
    sleep_duration = float(row['Sleep Duration'])
    quality_of_sleep = float(row['Quality of Sleep'])
    stress_level = int(row['Stress Level'])
    physical_activity = float(row['Physical Activity Level'])
    daily_steps = int(row['Daily Steps'])
    sleep_disorder = str(row['Sleep Disorder']).strip()

    if (
        sleep_duration < 6.0 or
        quality_of_sleep < 5 or
        stress_level >= 7 or
        physical_activity < 40 or
        daily_steps < 5000 or
        sleep_disorder != "None"
    ):
        label = 1  # Depressed
    else:
        label = 0  # Not Depressed

    labels.append(label)

# Add labels to DataFrame
df['Depression Label'] = labels

# Save output
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Labeled activity log saved to:", OUTPUT_PATH)
