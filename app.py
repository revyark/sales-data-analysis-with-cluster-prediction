from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load original scaled training data
X_train_scaled = joblib.load('X_train_scaled.pkl')

# Labels for clusters
cluster_labels = {
    0: 'Small Customer',
    1: 'Regular Customer',
    2: 'Bulk Buyer',
    3: 'High-Value Customer',
    4: 'Moderate Buyer',
    5: 'Low-Value Regular Customer',
    6: 'Occasional Customer',
    7: 'High-Value Occasional Buyer',
    8: 'Frequent Small Buyer',
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

    return render_template('result.html', cluster=cluster, meaning=meaning)

if __name__ == "__main__":
    app.run(debug=True)
