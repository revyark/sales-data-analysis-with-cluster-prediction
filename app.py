from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import firebase_admin
from firebase_admin import credentials, firestore

# Load Firebase credentials from environment variable
firebase_config = os.environ.get('FIREBASE_CONFIG_JSON')

# Parse the JSON string into a dictionary
import json
cred_dict = json.loads(firebase_config)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()

app = Flask(__name__)

# Load model and scaler
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load original scaled training data
X_train_scaled = joblib.load('X_train_scaled.pkl')

# Labels for clusters
cluster_labels = {
    0: 'High value Occasional Buyer',
    1: 'Regular Customer',
    2: 'Small CUstomer',
    3: 'Frequent Bulk Buyer',
    4: 'Moderate Buyer',
    5: 'Low-Value Regular Customer',
    6: 'Occasional Customer',
    7: 'Occasional Buyer',
    8: 'Occasional Bulk buyer',
    9: 'Inactive Customer'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    quantity = float(request.form['quantity'])
    orderline = float(request.form['orderline'])

    data = scaler.transform([[quantity, orderline]])
    cluster = model.predict(data)[0]
    meaning = cluster_labels.get(cluster, "Unknown")

    # Append user's data to training data
    X_augmented = np.vstack([X_train_scaled, data])
    updated_labels = model.predict(X_augmented)

    # Pie chart - updated counts
    counts = np.bincount(updated_labels)
    pie_labels = [cluster_labels.get(i, f'Cluster {i}') for i in range(len(counts))]
    colors = ['#00ffff', '#00c8ff', '#0096ff']

    from datetime import datetime

    from datetime import datetime


    # Dark theme and glow style
    plt.style.use('dark_background')
    sns.set(style="whitegrid", rc={'axes.facecolor':'#121212', 'figure.facecolor':'#121212'})



    # Pie chart
    plt.figure(figsize=(6, 6), facecolor='#121212')
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={'linewidth': 1, 'edgecolor': '#333'}
    )
    for w in wedges:
        w.set_alpha(0.8)
        w.set_edgecolor('white')
        w.set_linewidth(0.5)
    for text in texts + autotexts:
        text.set_color('white')
        text.set_fontsize(10)
    plt.tight_layout()
    plt.savefig('static/cluster_pie.png', transparent=True)
    plt.close()

    # Scatter plot (centroids and user input)
    centroids = model.cluster_centers_
    plt.figure(figsize=(8, 6), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')

    sns.scatterplot(
        x=centroids[:, 0], 
        y=centroids[:, 1], 
        s=300, 
        color='#00ffff', 
        marker='X', 
        label='Centroids',
        edgecolor='white',
        linewidth=0.5
    )

    sns.scatterplot(
        x=[data[0][0]], 
        y=[data[0][1]], 
        color='#ff00ff', 
        s=200, 
        label='Your Input',
        edgecolor='white',
        linewidth=0.5
    )

    plt.xlabel('Quantity (scaled)', color='white')
    plt.ylabel('Orderline (scaled)', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color('white')
    plt.tight_layout()
    plt.savefig('static/cluster_scatter.png', transparent=True)
    plt.close()

    # Histogram 1: Quantity (scaled)
    plt.figure(figsize=(8, 5), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')
    hist = sns.histplot(
        X_augmented[:, 0],
        bins=10,
        color="#00ffff",
        kde=True,
        edgecolor='white',
        linewidth=0.5
    )
    if hist.lines:
        hist.lines[0].set_color('#00c8ff')
        hist.lines[0].set_linewidth(2)
    plt.xlabel('Quantity (scaled)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.tight_layout()
    plt.savefig('static/hist_quantity.png', transparent=True)
    plt.close()

    # Histogram 2: Orderline (scaled)
    plt.figure(figsize=(8, 5), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')
    hist = sns.histplot(
        X_augmented[:, 1],
        bins=10,
        color="#00c8ff",
        kde=True,
        edgecolor='white',
        linewidth=0.5
    )
    if hist.lines:
        hist.lines[0].set_color('#0096ff')
        hist.lines[0].set_linewidth(2)
    plt.xlabel('Orderline (scaled)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.tight_layout()
    plt.savefig('static/hist_orderline.png', transparent=True)
    plt.close()

    

    # Store prediction result in Firebase Firestore
    db.collection('predictions').add({
        'quantity': quantity,
        'orderline': orderline,
        'cluster': int(cluster),
        'meaning': meaning,
        'timestamp': datetime.utcnow().isoformat()
    })
    return render_template('result.html', cluster=cluster, meaning=meaning)

@app.route('/dashboard')
def dashboard():
    # Retrieve all prediction documents
    predictions_ref = db.collection('predictions')
    docs = predictions_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()

    # Format documents into a list of dictionaries
    predictions = []
    for doc in docs:
        data = doc.to_dict()
        predictions.append({
            'quantity': data.get('quantity'),
            'orderline': data.get('orderline'),
            'cluster': data.get('cluster'),
            'meaning': data.get('meaning'),
            'timestamp': data.get('timestamp')
        })

    return render_template('dashboard.html', predictions=predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
