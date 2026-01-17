from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained ML model and scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
kmeans = joblib.load(os.path.join(base_dir, "model/kmeans_model.pkl"))
scaler = joblib.load(os.path.join(base_dir, "model/scaler.pkl"))

# Get cluster labels based on cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_avgs = [np.mean(c) for c in centers]
ranked = np.argsort(cluster_avgs)  # lowest to highest

# Map clusters to labels
labels = {
    ranked[0]: "Low Engagement Learner",
    ranked[1]: "Average Learner", 
    ranked[2]: "Active Learner"
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        attendance = float(request.form["attendance"])
        assignment = float(request.form["assignment"])
        quiz = float(request.form["quiz"])
        exam = float(request.form["exam"])

        # Scale input and predict using ML model
        data = np.array([[attendance, assignment, quiz, exam]])
        scaled = scaler.transform(data)
        cluster = kmeans.predict(scaled)[0]
        
        result = labels[cluster]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
