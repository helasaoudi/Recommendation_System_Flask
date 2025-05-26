import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
# Charger les variables d'environnement
load_dotenv()

# Télécharger les stop words en français si ce n'est pas déjà fait
nltk.download("stopwords")
stop_words_fr = stopwords.words("french")

# Initialiser Firestore
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    try:
        formation_id = request.args.get("formation_id")
        if not formation_id:
            return jsonify({"error": "formation_id is required"}), 400

        # Fetch all formations
        formations_ref = db.collection("Formation")
        formations = {doc.id: doc.to_dict() for doc in formations_ref.stream()}
        
        if not formations:
            return jsonify({"error": "No formations found"}), 404
            
        if formation_id not in formations:
            return jsonify({"error": "Formation not found"}), 404

        # Create DataFrame
        df = pd.DataFrame(formations).T
        
        # Handle missing fields with empty strings
        df["tag"] = df["tag"].fillna("")
        df["desc"] = df["desc"].fillna("")
        df["programme"] = df["programme"].fillna("")
        
        # Combine text fields
        df["text"] = df["tag"] + " " + df["desc"] + " " + df["programme"]
        
        # Remove empty text rows
        df = df[df["text"].str.strip() != ""]
        
        if len(df) < 2:
            return jsonify([])  # Not enough data for recommendations

        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words=stop_words_fr, max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(df["text"])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        df_similarity = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)
        
        # Get the 5 most similar formations (excluding the given one)
        if formation_id in df_similarity:
            similaires = df_similarity[formation_id].sort_values(ascending=False)[1:6]
            recommended_formation_ids = list(similaires.index)
        else:
            return jsonify([])
        
        # Return just the IDs (as your React code expects)
        return jsonify(recommended_formation_ids)
        
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)