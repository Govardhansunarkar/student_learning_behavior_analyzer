import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset
df = pd.read_csv(os.path.join(base_dir, "data/student.csv"))

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Train KMeans (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Save model and scaler
joblib.dump(kmeans, os.path.join(base_dir, "model/kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(base_dir, "model/scaler.pkl"))

# Show cluster centers (for understanding)
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=df.columns
)

print("Model trained and saved successfully!\n")
print("Cluster Centers:\n")
print(centers)
