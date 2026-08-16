# 🚢 Titanic AI Challenge — Live Prediction App

A no-code, live-prediction web app that predicts Titanic passenger survival in real time — built for the **Titanic AI Challenge** workshop, a 2-hour, zero-coding introduction to AI/ML.

Type in a passenger's age, class, gender, and fare, hit **Predict**, and watch a real trained machine learning model make the call instantly — with a confidence score and a breakdown of what drove the decision.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)

---

## ✨ What it does

- **Live prediction UI** — build a passenger with sliders/dropdowns (no code, no notebook) and get an instant survival prediction with a confidence percentage.
- **Quick-load presets** — the four passengers used in the workshop's opening icebreaker game, one click away, so you can compare the room's guesses against the model's.
- **Feature importance chart** — shows which factors (gender, fare, age, class...) actually drove the model's decision.
- **Model transparency** — displays real test accuracy and cross-validation scores instead of pretending the model is perfect.

## 🧠 The model

- **Algorithm:** `RandomForestClassifier` (scikit-learn)
- **Training data:** the classic [Titanic dataset](https://www.kaggle.com/c/titanic) — 891 passengers
- **Features used:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- **Test accuracy:** ~82%
- **5-fold cross-validation:** ~82% (± 3%)

## 📁 Repo structure

```
.
├── app.py               # Streamlit app (the UI)
├── train_model.py        # Trains the model and saves titanic_model.pkl
├── titanic_model.pkl     # Pre-trained model (ready to use out of the box)
├── titanic.csv            # Training data (891 passengers)
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Getting started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/titanic-ai-workshop.git
cd titanic-ai-workshop

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. No internet connection is required after the first install — handy for running it live in a room with unreliable Wi-Fi.

### Retraining the model

If you modify `titanic.csv` or want to tweak the model, regenerate `titanic_model.pkl`:

```bash
python train_model.py
```

## 🎤 Using it in a workshop

1. Open the app full-screen before your audience arrives, and test it once end-to-end.
2. Use the four preset buttons to replay the exact passengers from your opening "Would You Have Survived?" icebreaker — compare the model's call against the room's earlier vote.
3. Invite a volunteer to invent a new passenger; enter their answers live.
4. Point to the feature-importance chart to show *why* the model decided what it did — this is the moment that makes "machine learning" click for a non-technical audience.

## 🛠️ Built with

- [Streamlit](https://streamlit.io/) — the web app framework
- [scikit-learn](https://scikit-learn.org/) — model training
- [pandas](https://pandas.pydata.org/) — data handling

## 📄 License

MIT — free to use, adapt, and reuse for your own workshops or classrooms.
