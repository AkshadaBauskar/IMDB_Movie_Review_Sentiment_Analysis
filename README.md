# IMDB Movie Review Sentiment Analysis  

## Project Overview  
This project applies Deep Learning to the IMDB Large Movie Review Dataset, one of the most popular benchmarks in Natural Language Processing (NLP). The objective is to classify movie reviews as either positive or negative, enabling automated sentiment detection.  
The analysis leverages word embeddings and two deep learning architectures:  
- A **Multi-Layer Perceptron (MLP)** model with embeddings.  
- A **Convolutional Neural Network (CNN)** designed for sequential text data.  
By comparing these models, the project highlights how CNNs can better capture context and local word-order dependencies, leading to improved performance over traditional MLP approaches.  

## Problem Statement  
The IMDB dataset consists of:  
- 25,000 training reviews 
- 25,000 test reviews 
Each review is labeled as **positive (1)** or **negative (0)**.  
The challenge is a **binary sentiment classification task**:   Can a model learn to identify the sentiment from raw text sequences?  

## Dataset & Preprocessing  
- Dataset loaded directly from `keras.datasets.imdb`.  
- Vocabulary limited to the 5,000 most frequent words.  
- Reviews encoded into integers representing word frequency ranks.  
- Standardized sequence length to 500 tokens using padding/truncation (`pad_sequences`).  
- Dataset split into **training, validation, and test sets**.  

## Modeling Approach  

### 🔹 Model 1: Multi-Layer Perceptron (MLP)  
- Embedding layer: 500 words → 32-dimensional vectors  
- Flatten layer: Converts embeddings into a dense vector  
- Dense hidden layer: 256 ReLU units  
- Output layer: Sigmoid activation for binary classification  

📈 **Performance:** Test Accuracy ~ **86.04%**  

### 🔹 Model 2: Convolutional Neural Network (CNN)  
- Embedding layer: 32-dimensional embeddings  
- Conv1D layer: 32 filters, kernel size = 3  
- MaxPooling layer: pool size = 2  
- Dense hidden layer: 256 ReLU units  
- Output layer: Sigmoid activation  

📈 **Performance:** Test Accuracy ~ **88.82%**  

---

## 🔍 Training & Evaluation  
- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Batch Size:** 128  
- **Epochs:** 10 (with Early Stopping to prevent overfitting)  

Evaluation metrics included:  
- Accuracy  
- Training vs. Validation curves  
- Confusion matrix analysis  

---

## 📊 Results  

| Model                  | Test Accuracy |
|-------------------------|---------------|
| MLP (with embeddings)   | ~86.04%        |
| CNN (Conv1D)            | ~88.82%        |

✔️ The CNN model outperformed the MLP model by ~2.75%, confirming that convolutional filters help in capturing **n-gram patterns and context** in movie reviews.  
✔️ Both models demonstrated strong baseline performance, showcasing the **power of embeddings** over simple bag-of-words approaches.  

---

## ⚙️ Tech Stack  
- **Language:** Python  
- **Deep Learning Frameworks:** Keras, TensorFlow  
- **Libraries:** NumPy, Matplotlib, Seaborn  
- **Techniques:** Word Embeddings, MLP, CNN, Early Stopping  

---

## 🚀 How to Run  

Clone the repository:  
```bash
git clone https://github.com/AkshadaBauskar/IMDB_Movie_Review_Sentiment_Analysis.git
cd IMDB_Movie_Review_Sentiment_Analysis
