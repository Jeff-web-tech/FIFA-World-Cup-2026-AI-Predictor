# ⚽ FIFA World Cup 2026 AI Predictor

An AI-powered football prediction system built using Machine Learning, Flask, Python, and SQLite.

The system analyzes historical international football matches and predicts the outcome of upcoming FIFA World Cup 2026 matches using advanced football metrics such as:

- Team Form
- Goals Scored
- Goals Conceded
- Expected Goals (xG)
- Team Strength Ratings (ELO-style)
- Tournament Importance
- FIFA Rankings

---

# 🚀 Features

✅ Match Outcome Prediction

Predicts:

- Home Win (H)
- Draw (D)
- Away Win (A)

---

✅ Probability Analysis

Displays:

- Home Win Probability
- Draw Probability
- Away Win Probability

---

✅ Historical Tracking

Stores all predictions inside a SQLite database.

Users can view:

- Previous predictions
- Prediction confidence
- Match history

---

✅ FIFA Ranking Integration

The model includes FIFA rankings to improve prediction quality and capture team strength more accurately.

---

✅ Interactive Web Interface

Built with:

- HTML
- CSS
- JavaScript
- Flask

Users can:

- Select teams
- View group fixtures
- Run predictions instantly

---

# 🧠 Machine Learning Pipeline

## Data Source

Historical international football results dataset.

Main fields:

- Date
- Home Team
- Away Team
- Home Score
- Away Score
- Tournament
- Neutral Venue

---

## Feature Engineering

The model creates football-specific features:

### Team Form

Points earned from the last 5 matches.

### Goals For

Average goals scored in the last 5 matches.

### Goals Against

Average goals conceded in the last 5 matches.

### Expected Goals (xG)

Estimated attacking performance.

### Team Strength

Custom ELO-style rating system.

### Tournament Importance

Different weights assigned to tournaments:

| Tournament | Weight |
|------------|---------|
| FIFA World Cup | 5 |
| UEFA Euro | 4 |
| Copa América | 4 |
| AFCON | 3 |
| AFC Asian Cup | 3 |

---

## Model Training

Models experimented with:

- Random Forest Classifier
- Logistic Regression
- Gradient Boosting
- Ensemble Voting Models

Final model trained using Scikit-Learn.

---

# 🏗️ Tech Stack

## Backend

- Python
- Flask
- Scikit-Learn
- Pandas
- NumPy
- SQLite

## Frontend

- HTML
- CSS
- JavaScript

## Deployment

- GitHub
- Render

---

# 📂 Project Structure

```text
FIFA-World-Cup-2026-AI-Predictor/
│
├── app.py
├── worldcup_ai.py
├── model.pkl
├── scaler.pkl
├── engineered_data.csv
├── fifa_rankings.csv
├── predictions.db
├── requirements.txt
├── Procfile
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/Jeff-web-tech/FIFA-World-Cup-2026-AI-Predictor.git
```

Move into project:

```bash
cd FIFA-World-Cup-2026-AI-Predictor
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# 📊 Example Prediction

### Match

Argentina 🇦🇷 vs France 🇫🇷

### Output

```text
Prediction: Home Win

Home Win Probability: 54.3%
Draw Probability: 23.8%
Away Win Probability: 21.9%
```

---

# 🔮 Future Improvements

Planned upgrades:

- Live FIFA Rankings API
- Live Match Data API
- Tournament Simulation Engine
- Full 2026 World Cup Bracket Prediction
- Player Statistics Integration
- Injury Tracking
- Team News Analysis
- Deep Learning Models
- XGBoost Integration
- Mobile Application

---

# 📚 Skills Demonstrated

This project demonstrates practical experience in:

- Machine Learning
- Data Cleaning
- Feature Engineering
- Classification Models
- Model Evaluation
- Flask Development
- REST APIs
- Database Design
- SQL
- Frontend Development
- Git & GitHub
- Deployment
- Software Engineering

---

# 👨‍💻 Author

**Jeffery Anyimah**

Computer Science Student  
University of Ghana

LinkedIn: www.linkedin.com/in/jeffery-anyimah-356422370

GitHub: https://github.com/Jeff-web-tech

---

# ⭐ Support

If you found this project interesting, consider giving it a star on GitHub.
