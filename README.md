# Course Recommendation Engine: From Content-Based to Neural Collaborative Filtering

This repository hosts a comprehensive end-to-end machine learning pipeline designed to solve the problem of information overload on large-scale online learning platforms. By analyzing 33,901 user-course interactions across 126 technical courses, this system transitions from simple content-based filtering to advanced **Neural Network Embeddings** to deliver high-precision, personalized course suggestions.

## 📊 Dataset & EDA Insights
The underlying data follows a classic **long-tail distribution**, where most users enroll in only 1–10 courses, creating a significant data sparsity challenge.

* **Domain Focus:** Course content is heavily concentrated in Backend Development, Machine Learning, and Database Management.
* **Engagement Drivers:** Foundations courses like *Python for Data Science* and *Big Data 101* dominate user interactions, serving as primary "hubs" for learning paths.
* **Keywords:** A word cloud analysis confirms a technical stack centered on Python, SQL, Cloud Computing, and Microservices.

## 📁 Project Structure

The project is organized into two primary phases, documenting the evolution from unsupervised similarity models to supervised deep learning architectures.

### I. Content-Based Recommender System (Unsupervised Learning)
* **User Profile and Course Genres:** Engineering user preference vectors based on historical course category engagement.
* **Course Similarity:** Implementation of geometric comparison metrics (Cosine Similarity and Jaccard Index) to recommend courses with similar technical content.
* **Clustering-Based Recommendations:** Utilizing K-Means clustering on PCA-reduced data to segment users into 10 distinct "Learner Personas" for group-based discovery.

### II. Collaborative-Filtering Recommender System (Supervised Learning)
* **KNN-Based Modeling:** Establishing a baseline using K-Nearest Neighbors to predict ratings based on user-item proximity.
* **NMF-Based Modeling:** Implementing Non-Negative Matrix Factorization with 32 latent factors to decompose the interaction matrix.
* **Neural Network Modeling:** A Deep Learning approach using dual embedding layers to represent users and items in a shared latent space.
* **Regression Embedding:** Predicting explicit rating values using a continuous output layer.
* **Classification Embedding:** An experimental branch focusing on predicting the probability of enrollment.

## 🛠️ The Recommendation Pipeline
The project implements a multi-model hybrid approach to evaluate which architecture best captures latent user preferences.

### 1. Content & Clustering (Unsupervised)
* **User Profile Vectors:** Recommends courses based on an interest threshold (Score $\ge$ 40).
* **Course Similarity:** Uses geometric comparison (Cosine/Jaccard) with a 0.5 similarity threshold.
* **Group-Based Discovery:** Leverages K-Means clustering on PCA-reduced data (reducing 15D to 9D while retaining 90% variance).

### 2. Collaborative Filtering (Supervised)
* **NMF (Baseline):** Utilized Non-Negative Matrix Factorization with 32 latent factors as a traditional benchmark.
* **The Pivot:** Initial attempts at non-linear classification (Random Forest/SVM) failed to outperform a random-guess baseline (33.3%) due to data sparsity.
* **Neural Collaborative Filtering (NCF):** Developed a deep learning model using embedding layers and ReLU activation to model user-item interactions.

## 🏆 Performance Evaluation
The Neural Network emerged as the gold standard, significantly reducing prediction error by capturing relationships that linear models missed.

| Model | Evaluation Metric | Result | Error Reduction |
| :--- | :--- | :--- | :--- |
| **KNN (Baseline)** | RMSE | 1.2923 | - |
| **NMF** | RMSE | 1.2861 | Baseline |
| **Neural Network** | **RMSE** | **0.4437** | **~65% vs KNN** |

## 🚀 Tech Stack & Deployment
* **Logic:** Python, Pandas, Scikit-Learn.
* **Deep Learning:** TensorFlow/Keras (Embeddings, Adam Optimizer, MSE Loss).
* **Clustering:** PCA & K-Means for user segmentation.
* **Deployment:** The system is designed for real-time recommendations via a **Streamlit** dashboard.

## 📬 Contact
**Seiha Vat** *Data Scientist* [LinkedIn Profile](www.linkedin.com/in/seiha-vat-49014b242) 
