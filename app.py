from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("recommendation_model.pkl", "rb"))

# Load data
user_item_matrix = pd.read_csv(r"C:\Users\athun\Downloads\hello\user_item_matrix (3) (1).csv", index_col=0)

def recommend_products(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_items = user_item_matrix.loc[user_id]
    unseen_items = user_items[user_items == 0].index

    predictions = [(item, model.predict(user_id, item).est) for item in unseen_items]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:top_n]

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None

    if request.method == "POST":
        try:
            user_id = int(request.form["user_id"])
            recommendations = recommend_products(user_id)
        except:
            recommendations = []

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
