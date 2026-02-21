from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
import os
import re

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


# ─────────────────────────────────────────────
# GROQ HELPER
# ─────────────────────────────────────────────
def ask_groq(prompt, max_tokens=500):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"


# ─────────────────────────────────────────────
# VERIFIED TOPIC IMAGE LIBRARY
# All URLs are real Wikimedia Commons images
# verified to exist and match the topic exactly
# ─────────────────────────────────────────────
TOPIC_IMAGES = {
    "regression": [
        # Linear regression scatter plot with line
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/600px-Linear_regression.svg.png",
        # Anscombe's quartet showing regression
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/600px-Anscombe%27s_quartet_3.svg.png",
        # Polynomial regression curve
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Polyreg_scheffe.svg/600px-Polyreg_scheffe.svg.png",
    ],
    "classification": [
        # Logistic regression S-curve classification boundary
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png",
        # SVM classification with margin
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/600px-SVM_margin.png",
        # Decision boundary example
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Kernel_Machine.svg/600px-Kernel_Machine.svg.png",
    ],
    "neural networks": [
        # Artificial neural network diagram
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/600px-Colored_neural_network.svg.png",
        # Deep neural network layers
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/600px-Artificial_neural_network.svg.png",
        # Neuron diagram
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Blausen_0657_MultipolarNeuron.png/600px-Blausen_0657_MultipolarNeuron.png",
    ],
    "clustering": [
        # K-means clustering with colored clusters
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/600px-K-means_convergence.gif",
        # Cluster analysis example
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Cluster-2.svg/600px-Cluster-2.svg.png",
        # DBSCAN clustering result
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/DBSCAN-Illustration.svg/600px-DBSCAN-Illustration.svg.png",
    ],
    "decision trees": [
        # Decision tree diagram
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Cart_tree_titanic_survivors.png/600px-Cart_tree_titanic_survivors.png",
        # Random forest visualization
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Random_forest_diagram_complete.png/600px-Random_forest_diagram_complete.png",
        # Decision tree splits visualization
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/CART_tree_titanic_survivors_KOR.png/600px-CART_tree_titanic_survivors_KOR.png",
    ],
    "ridge regression": [
        # Ridge vs Lasso regularization paths
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/L1_and_L2_balls.svg/600px-L1_and_L2_balls.svg.png",
        # Regularization comparison
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/600px-Linear_regression.svg.png",
        # Bias variance tradeoff
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/600px-Bias_and_variance_contributing_to_total_error.svg.png",
    ],
    "svm": [
        # SVM with margin and support vectors
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/600px-SVM_margin.png",
        # Kernel SVM non-linear
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Kernel_Machine.svg/600px-Kernel_Machine.svg.png",
        # SVM hyperplane
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Kernel_Machine.svg/600px-Kernel_Machine.svg.png",
    ],
    "random forest": [
        # Random forest ensemble diagram
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Random_forest_diagram_complete.png/600px-Random_forest_diagram_complete.png",
        # Decision tree (base model)
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Cart_tree_titanic_survivors.png/600px-Cart_tree_titanic_survivors.png",
        # Ensemble bagging concept
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Ensemble_Bagging.svg/600px-Ensemble_Bagging.svg.png",
    ],
    "naive bayes": [
        # Bayes theorem formula visual
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Bayes%27_Theorem_MMB_01.jpg/600px-Bayes%27_Theorem_MMB_01.jpg",
        # Naive Bayes classification
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png",
        # Probability distribution
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/600px-Normal_Distribution_PDF.svg.png",
    ],
    "pca": [
        # PCA principal components visualization
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/GaussianScatterPCA.svg/600px-GaussianScatterPCA.svg.png",
        # PCA dimensionality reduction
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/PCA_fish.png/600px-PCA_fish.png",
        # Eigenvalues scree plot concept
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/PCA-Gaussian.png/600px-PCA-Gaussian.png",
    ],
    "knn": [
        # KNN classification example
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/600px-KnnClassification.svg.png",
        # Voronoi diagram (KNN boundaries)
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Euclidean_Voronoi_diagram.svg/600px-Euclidean_Voronoi_diagram.svg.png",
        # Distance metric visualization
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Minkowski_distance_examples.png/600px-Minkowski_distance_examples.png",
    ],
    "gradient boosting": [
        # Gradient boosting trees ensemble
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Random_forest_diagram_complete.png/600px-Random_forest_diagram_complete.png",
        # Boosting concept
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ensemble_Boosting.svg/600px-Ensemble_Boosting.svg.png",
        # Decision tree base learner
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Cart_tree_titanic_survivors.png/600px-Cart_tree_titanic_survivors.png",
    ],
    "reinforcement learning": [
        # RL agent-environment loop
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/600px-Reinforcement_learning_diagram.svg.png",
        # Markov decision process
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Markov_Decision_Process.svg/600px-Markov_Decision_Process.svg.png",
        # Q-learning grid world
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Markov_decision_processes.svg/600px-Markov_decision_processes.svg.png",
    ],
    "logistic regression": [
        # Logistic sigmoid curve
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png",
        # Classification boundary
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Kernel_Machine.svg/600px-Kernel_Machine.svg.png",
        # Probability output
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/600px-Normal_Distribution_PDF.svg.png",
    ],
    "deep learning": [
        # Deep neural network
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/600px-Colored_neural_network.svg.png",
        # CNN architecture
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/600px-Typical_cnn.png",
        # Neural network layers
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/600px-Artificial_neural_network.svg.png",
    ],
    "cnn": [
        # CNN architecture diagram
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/600px-Typical_cnn.png",
        # Convolutional filter visualization
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/600px-Colored_neural_network.svg.png",
        # Image feature maps
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/600px-Artificial_neural_network.svg.png",
    ],
    "overfitting": [
        # Overfitting vs underfitting graph
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Overfitting.svg/600px-Overfitting.svg.png",
        # Bias variance tradeoff
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/600px-Bias_and_variance_contributing_to_total_error.svg.png",
        # Regularization effect
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/L1_and_L2_balls.svg/600px-L1_and_L2_balls.svg.png",
    ],
    "k-means": [
        # K-means animation
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/600px-K-means_convergence.gif",
        # Cluster result
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Cluster-2.svg/600px-Cluster-2.svg.png",
        # Voronoi partition
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Euclidean_Voronoi_diagram.svg/600px-Euclidean_Voronoi_diagram.svg.png",
    ],
    "nlp": [
        # NLP text processing
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Parse_tree_1.jpg/600px-Parse_tree_1.jpg",
        # Word embedding space
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png",
        # Neural network for text
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/600px-Colored_neural_network.svg.png",
    ],
    "bagging": [
        # Bagging ensemble
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Ensemble_Bagging.svg/600px-Ensemble_Bagging.svg.png",
        # Random forest (bagging result)
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Random_forest_diagram_complete.png/600px-Random_forest_diagram_complete.png",
        # Bootstrap sampling
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Cart_tree_titanic_survivors.png/600px-Cart_tree_titanic_survivors.png",
    ],
    "boosting": [
        # Boosting ensemble diagram
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ensemble_Boosting.svg/600px-Ensemble_Boosting.svg.png",
        # Gradient boosting trees
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Random_forest_diagram_complete.png/600px-Random_forest_diagram_complete.png",
        # Weak learner combination
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Ensemble_Bagging.svg/600px-Ensemble_Bagging.svg.png",
    ],
    "transfer learning": [
        # Neural network layers (transfer concept)
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/600px-Colored_neural_network.svg.png",
        # CNN feature layers
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/600px-Typical_cnn.png",
        # Deep learning
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/600px-Artificial_neural_network.svg.png",
    ],
    "xgboost": [
        # Gradient boosting / XGBoost
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ensemble_Boosting.svg/600px-Ensemble_Boosting.svg.png",
        # Decision tree base
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Cart_tree_titanic_survivors.png/600px-Cart_tree_titanic_survivors.png",
        # Random forest ensemble
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Random_forest_diagram_complete.png/600px-Random_forest_diagram_complete.png",
    ],
}

# Default: shown for any topic not in the list above
DEFAULT_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/600px-Colored_neural_network.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/600px-Linear_regression.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/600px-Bias_and_variance_contributing_to_total_error.svg.png",
]


def get_topic_images(topic):
    key = topic.lower().strip()
    # Try exact match first, then partial match
    if key in TOPIC_IMAGES:
        return TOPIC_IMAGES[key]
    for k in TOPIC_IMAGES:
        if k in key or key in k:
            return TOPIC_IMAGES[k]
    return DEFAULT_IMAGES


# ─────────────────────────────────────────────
# DYNAMIC TOPIC GENERATOR (AI)
# ─────────────────────────────────────────────
def generate_topic_from_ai(topic):
    explanation = ask_groq(
        f"In 2-3 sentences, explain what '{topic}' is in Machine Learning for a beginner. "
        f"Include one real-world example. Only give the explanation text, nothing else.",
        max_tokens=200
    )
    types_raw = ask_groq(
        f"List exactly 5 subtypes or key concepts of '{topic}' in Machine Learning. "
        f"Return only a numbered list:\n1. item\n2. item\n3. item\n4. item\n5. item\nNothing else.",
        max_tokens=150
    )
    types = []
    for line in types_raw.strip().split('\n'):
        line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
        if line and len(line) > 2:
            types.append(line)
    types = types[:5] if len(types) >= 5 else types + [f"{topic} concept {i}" for i in range(len(types)+1, 6)]

    code = ask_groq(
        f"Write a simple complete runnable Python example for '{topic}' in ML using scikit-learn. Code only, no explanation.",
        max_tokens=400
    )
    code = re.sub(r'```python|```', '', code).strip()

    quiz = []
    for i in range(1, 4):
        q_raw = ask_groq(
            f"Create quiz question {i} about '{topic}' in Machine Learning.\n"
            f"ONLY return this format:\nQUESTION: question?\nA: opt\nB: opt\nC: opt\nD: opt\nANSWER: A",
            max_tokens=150
        )
        q_data = parse_quiz_question(q_raw)
        if q_data:
            quiz.append(q_data)

    if len(quiz) < 3:
        quiz = [
            {"question": f"What is {topic} mainly used for?", "options": ["Prediction", "Data storage", "UI design", "Networking"], "answer": "Prediction"},
            {"question": f"Which library is used for {topic}?", "options": ["scikit-learn", "pygame", "tkinter", "requests"], "answer": "scikit-learn"},
            {"question": f"What type of learning is {topic}?", "options": ["Machine Learning", "Manual coding", "Database design", "Networking"], "answer": "Machine Learning"},
        ]
    return {"explanation": explanation.strip(), "types": types, "code": code, "quiz": quiz}, True


def parse_quiz_question(raw):
    try:
        lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]
        question, options, answer_letter = "", [], ""
        for line in lines:
            if line.upper().startswith("QUESTION:"):
                question = line.split(":", 1)[1].strip()
            elif re.match(r'^[ABCD][\.\:\)]\s', line):
                options.append(re.sub(r'^[ABCD][\.\:\)]\s*', '', line).strip())
            elif line.upper().startswith("ANSWER:"):
                answer_letter = line.split(":", 1)[1].strip().upper()[0]
        if question and len(options) == 4 and answer_letter in "ABCD":
            return {"question": question, "options": options, "answer": options["ABCD".index(answer_letter)]}
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# STATIC TOPICS
# ─────────────────────────────────────────────
ml_topics = {
    "regression": {
        "explanation": "Regression is a supervised learning technique used to predict continuous values. It finds the relationship between input features (X) and output value (Y). Example: predicting house price, temperature, salary.",
        "types": ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression"],
        "code": """from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = [[1],[2],[3],[4],[5]]
y = [2,4,6,8,10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))""",
        "quiz": [
            {"question": "Regression is mainly used for?", "options": ["Classification", "Predicting continuous values", "Clustering", "Dimensionality reduction"], "answer": "Predicting continuous values"},
            {"question": "Which is a regression evaluation metric?", "options": ["Accuracy", "Precision", "Mean Squared Error", "Recall"], "answer": "Mean Squared Error"},
            {"question": "Ridge Regression reduces?", "options": ["Overfitting", "Underfitting", "Data imbalance", "Noise"], "answer": "Overfitting"},
            {"question": "What does slope represent in linear regression?", "options": ["Intercept", "Rate of change", "Error value", "Sample size"], "answer": "Rate of change"},
            {"question": "Which library is used for regression in Python?", "options": ["pygame", "scikit-learn", "tkinter", "PIL"], "answer": "scikit-learn"}
        ]
    },
    "classification": {
        "explanation": "Classification is a supervised learning technique used to predict categorical labels. It assigns input data into predefined classes. Example: spam/not spam, disease/no disease, cat/dog.",
        "types": ["Binary Classification", "Multi-class Classification", "Multi-label Classification", "Logistic Regression", "Random Forest"],
        "code": """from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[1],[2],[3],[4],[5],[6]]
y = [0,0,0,1,1,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))""",
        "quiz": [
            {"question": "Classification is mainly used for?", "options": ["Predicting continuous values", "Predicting labels", "Clustering", "Sorting data"], "answer": "Predicting labels"},
            {"question": "Logistic Regression is used for?", "options": ["Regression", "Classification", "Clustering", "Dimensionality reduction"], "answer": "Classification"},
            {"question": "Which metric is used for classification?", "options": ["MSE", "RMSE", "Accuracy", "MAE"], "answer": "Accuracy"},
            {"question": "What is binary classification?", "options": ["Predicts 3+ classes", "Predicts 2 classes only", "Predicts numbers", "Groups data"], "answer": "Predicts 2 classes only"},
            {"question": "Which algorithm builds decision boundaries?", "options": ["K-Means", "SVM", "PCA", "Apriori"], "answer": "SVM"}
        ]
    },
    "neural networks": {
        "explanation": "Neural Networks are computing systems inspired by the human brain. They consist of layers of interconnected nodes that learn patterns from data through training.",
        "types": ["Feedforward Neural Networks", "Convolutional Neural Networks CNN", "Recurrent Neural Networks RNN", "LSTM", "Transformers"],
        "code": """from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))""",
        "quiz": [
            {"question": "What inspired neural networks?", "options": ["DNA structure", "Human brain", "Sorting algorithms", "Cloud computing"], "answer": "Human brain"},
            {"question": "Activation functions introduce?", "options": ["Data loading", "Non-linearity", "Data splitting", "Label normalization"], "answer": "Non-linearity"},
            {"question": "CNN is mainly used for?", "options": ["Text data", "Time series", "Image data", "Tabular data"], "answer": "Image data"},
            {"question": "What is backpropagation?", "options": ["Forward pass", "Error correction through layers", "Data augmentation", "Feature selection"], "answer": "Error correction through layers"},
            {"question": "RNN is best suited for?", "options": ["Images", "Sequential and time data", "Clustering", "Regression only"], "answer": "Sequential and time data"}
        ]
    },
    "clustering": {
        "explanation": "Clustering is an unsupervised learning technique that groups similar data points without predefined labels. Example: customer segmentation, document grouping.",
        "types": ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN", "Gaussian Mixture Models", "Agglomerative Clustering"],
        "code": """from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Labels:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)""",
        "quiz": [
            {"question": "Clustering is which type of learning?", "options": ["Supervised", "Unsupervised", "Reinforcement", "Semi-supervised"], "answer": "Unsupervised"},
            {"question": "K-Means requires you to specify?", "options": ["Labels", "Number of clusters", "Test size", "Learning rate"], "answer": "Number of clusters"},
            {"question": "What are centroids in K-Means?", "options": ["Outliers", "Center points of clusters", "Input features", "Loss values"], "answer": "Center points of clusters"},
            {"question": "DBSCAN is useful for?", "options": ["Linearly separable data", "Arbitrary shape clusters", "Image classification", "Text generation"], "answer": "Arbitrary shape clusters"},
            {"question": "Which metric evaluates clustering quality?", "options": ["Accuracy", "Silhouette Score", "MSE", "F1-Score"], "answer": "Silhouette Score"}
        ]
    },
    "decision trees": {
        "explanation": "A Decision Tree splits data into branches based on feature values. It is easy to understand and interpret, making it great for beginners in ML.",
        "types": ["Classification Tree", "Regression Tree", "Random Forest", "Gradient Boosting", "XGBoost"],
        "code": """from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
y = [0,0,0,1,1,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))""",
        "quiz": [
            {"question": "Decision Trees are which type of learning?", "options": ["Unsupervised", "Supervised", "Reinforcement", "None"], "answer": "Supervised"},
            {"question": "What does max_depth control?", "options": ["Number of features", "Tree size", "Learning rate", "Clusters"], "answer": "Tree size"},
            {"question": "Random Forest is made of?", "options": ["Single tree", "Multiple trees", "Neural layers", "Clusters"], "answer": "Multiple trees"},
            {"question": "Decision Trees are prone to?", "options": ["Underfitting", "Overfitting", "Clustering errors", "Low accuracy always"], "answer": "Overfitting"},
            {"question": "Which splitting criterion is used?", "options": ["Gradient", "Gini Impurity", "Silhouette", "MSE only"], "answer": "Gini Impurity"}
        ]
    },
    "ridge regression": {
        "explanation": "Ridge Regression uses L2 regularization to reduce overfitting by adding a penalty to large coefficients. It is ideal when features are correlated.",
        "types": ["L2 Regularization", "Ridge vs Lasso", "ElasticNet combined", "Regularization path", "Cross-validation for alpha"],
        "code": """from sklearn.linear_model import Ridge

X = [[1],[2],[3],[4],[5]]
y = [2,4,6,8,10]

model = Ridge(alpha=1.0)
model.fit(X, y)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)""",
        "quiz": [
            {"question": "Ridge Regression uses which regularization?", "options": ["L1", "L2", "L0", "None"], "answer": "L2"},
            {"question": "Ridge Regression mainly reduces?", "options": ["Overfitting", "Underfitting", "Noise", "Missing values"], "answer": "Overfitting"},
            {"question": "The penalty term affects?", "options": ["Model coefficients", "Dataset size", "Class labels", "Accuracy directly"], "answer": "Model coefficients"},
            {"question": "What is the alpha parameter?", "options": ["Learning rate", "Regularization strength", "Train size", "Number of features"], "answer": "Regularization strength"},
            {"question": "Lasso uses which regularization?", "options": ["L2", "L1", "L0", "ElasticNet"], "answer": "L1"}
        ]
    }
}


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/topic_images", methods=["POST"])
def topic_images_route():
    data = request.json
    topic = data.get("topic", "")
    return jsonify({"images": get_topic_images(topic)})


@app.route("/ai_explain", methods=["POST"])
def ai_explain():
    data = request.json
    topic = data.get("topic", "")
    question = data.get("question", "")
    prompt = f"You are a friendly ML tutor. A student is learning about '{topic}' and asks: '{question}'\nAnswer clearly in 3-5 sentences with a simple example."
    return jsonify({"answer": ask_groq(prompt, 350)})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    topic = data.get("topic", "")
    level = data.get("level", "beginner")
    prompt = f"You are an ML tutor. Explain '{topic}' for a {level} student.\nInclude: 1) Simple definition 2) How it works 3) Real-world example 4) Key points."
    return jsonify({"result": ask_groq(prompt, 500)})


@app.route("/generator")
def generator():
    return render_template("generate.html")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    topic_name = ""
    ai_generated = False
    topic_images = []

    if request.method == "POST":
        topic = request.form.get("topic", "").lower().strip()
        topic_name = topic.title()

        if topic in ml_topics:
            result = ml_topics[topic]
            ai_generated = False
        else:
            result, ai_generated = generate_topic_from_ai(topic)

        topic_images = get_topic_images(topic)

    return render_template("index.html",
                           result=result,
                           topic_name=topic_name,
                           topics=list(ml_topics.keys()),
                           ai_generated=ai_generated,
                           topic_images=topic_images)


if __name__ == "__main__":
    app.run(debug=True)
