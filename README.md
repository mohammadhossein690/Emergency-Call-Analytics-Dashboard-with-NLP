# 🚨 Emergency Call Analytics Dashboard with NLP

An interactive **Streamlit dashboard** for analyzing 911 emergency calls and predicting emergency categories using **Natural Language Processing**.

## Features

- 📊 Emergency call EDA and visualization
- 🌎 Geographic hotspot analysis
- ⏰ Time-based pattern analysis
- 🤖 NLP text classification model
- 🔮 Live emergency reason prediction

## Machine Learning

**Model:** Multinomial Naive Bayes  
**Features:** CountVectorizer + Text Preprocessing

**Classes:**

- EMS
- Fire
- Traffic

**Performance:**

- Accuracy: 93.10%

## Tech Stack

Python • Streamlit • Pandas • Plotly • Scikit-learn • NLTK

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
