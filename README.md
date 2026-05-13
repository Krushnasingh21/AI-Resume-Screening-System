# 📄 Resume Screening & Classification

An automated resume screening system that uses Natural Language Processing (NLP) and Machine Learning to classify resumes into 24 different job categories. The project employs TF-IDF vectorization and K-Nearest Neighbors (KNN) classification to analyze resume text and predict the most suitable job category.

---

## 📁 Repository Structure

```
resume-screening/
│
├── data/
│   └── resume_dataset.csv        # Dataset with 2,484 resumes across 24 job categories
│
├── notebooks/
│   └── Resume_Screening.ipynb    # Main notebook (EDA, preprocessing, modeling)
│
└── README.md
```

---

## 📊 Dataset

**File:** `data/resume_dataset.csv`

A collection of **2,484 resume records** spanning **24 job categories**, originally sourced from job portals.

| Column | Description |
|---|---|
| `ID` | Unique identifier for each resume |
| `Resume_str` | Plain text version of the resume |
| `Resume_html` | HTML-formatted version of the resume |
| `Category` | Job category label (target variable) |

### Job Categories (24 total)

The dataset covers diverse industries and roles:

- **Information Technology** (120 resumes)
- **Business Development** (120)
- **Engineering** (118)
- **Finance, Accountant, Banking** (351 combined)
- **HR, Advocate, Chef** (346 combined)
- **Healthcare, Fitness, Consultant** (347 combined)
- **Sales, Teacher, Designer** (325 combined)
- **Construction, Public Relations, Arts** (326 combined)
- **Aviation, Apparel, Digital Media** (310 combined)
- **Agriculture** (63)
- **Automobile** (36)
- **BPO** (22)

> **Note:** The dataset is slightly imbalanced, with Information Technology and Business Development having the most samples, while BPO has the fewest.

---

## 🔬 Methodology

The notebook `Resume_Screening.ipynb` implements the following pipeline:

1. **Data Loading & Exploration** — Overview of resume categories and distribution
2. **Text Cleaning** — Preprocessing resume text:
   - Remove URLs, hashtags, mentions, punctuation
   - Remove non-ASCII characters
   - Remove extra whitespace
3. **Exploratory Data Analysis (EDA)** 
   - Category distribution visualization (count plot)
   - Word frequency analysis (top 50 most common words)
   - Word cloud generation for visual insights
4. **Feature Engineering**
   - Label encoding for categories
   - TF-IDF vectorization with max 1,500 features
   - Stop word removal (English stopwords)
5. **Train-Test Split** — 80% training / 20% testing with `random_state=0`
6. **Model Training** — K-Nearest Neighbors (KNN) with OneVsRestClassifier
7. **Evaluation** — Classification report with precision, recall, and F1-score per category

---

## 📈 Model Results

**Algorithm:** K-Nearest Neighbors (KNN) with OneVsRestClassifier

| Metric | Training Set | Test Set |
|---|---|---|
| **Accuracy** | 66% | 50% |

### Per-Category Performance (Test Set)

Best-performing categories (F1-score > 0.70):
- **Information Technology** (0.79)
- **Engineering** (0.73)
- **Sales** (0.71)
- **Healthcare** (0.71)

Challenging categories (F1-score < 0.20):
- **Agriculture** (0.00) — Limited training samples (63)
- **Automobile** (0.00) — Limited training samples (36)
- **Advocate** (0.09)
- **Business Development** (0.11)

> **Observations:** The model shows moderate accuracy but struggles with underrepresented categories and similar job roles. The 16% accuracy drop from training to test suggests some overfitting. Further improvements could include:
> - Balancing the dataset (upsampling/downsampling)
> - Testing other algorithms (Random Forest, SVM, Neural Networks)
> - Hyperparameter tuning for KNN (number of neighbors, distance metrics)
> - Expanding the feature set beyond 1,500 TF-IDF features

---

## 🔍 Key Insights from EDA

Top words in resumes (across all categories):
- **HR**: 1,045 occurrences
- **State, City, Company**: High frequency of location/organization names
- **Management, Employee, Training**: Common professional terms
- **Skills, Experience, Education**: Resume structure keywords

---

## ⚙️ Requirements

```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
```

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
```

**Additional NLTK Data:**

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/resume-screening.git
   cd resume-screening
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
   ```

3. **Download NLTK data** (run in Python)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

4. **Launch the notebook**
   ```bash
   jupyter notebook notebooks/Resume_Screening.ipynb
   ```

---

## 📌 Notes

- The notebook was originally developed on Google Colab. If running locally, update the data path in the notebook from `/content/resume_dataset.csv` to `../data/resume_dataset.csv`.
- The dataset file is **54 MB** — ensure you have Git LFS configured if pushing to GitHub, or host it externally (Kaggle, Google Drive) and link it in the README.
- The word cloud generation processes only the first 160 resumes for performance reasons. You can modify the range in the code to include all resumes.

---

## 🚧 Future Improvements

- [ ] Address class imbalance with SMOTE or class weighting
- [ ] Experiment with advanced models (Random Forest, XGBoost, BERT embeddings)
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Build a web interface for real-time resume classification
- [ ] Extract and visualize key skills per job category
- [ ] Add support for additional resume formats (PDF, DOCX)

---

## 📄 License

This project is open-source and available under the MIT License.
