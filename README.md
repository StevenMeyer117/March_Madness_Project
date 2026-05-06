# 🏀 March Madness Predictor

## Overview

This project is a web-based application that simulates the NCAA March Madness basketball tournament using a machine learning model and Monte Carlo simulation. The goal is to predict game outcomes, estimate championship probabilities, and visualize tournament progression in an interactive format.

Users can run multiple simulations to explore how likely teams are to advance through each round and ultimately win the tournament.

---

## Stakeholder / Use Case

This application is useful for:

- College basketball fans interested in bracket predictions  
- Analysts exploring team performance and matchup outcomes  
- Students learning about machine learning and simulation techniques  
- Casual users who want a data-driven bracket  

The tool provides insights into tournament variability and highlights how different teams perform across many simulated outcomes.

---

## Data Description

The dataset used in this project contains historical college basketball team statistics, including:

- Offensive efficiency (ADJOE)  
- Defensive efficiency (ADJDE)  
- Win percentage (WP)  
- Strength metrics (BARTHAG, WAB)  
- Turnover rates, rebounding, shooting percentages, and more  

A derived feature called **NET_EFF (Net Efficiency)** is computed as:


NET_EFF = ADJOE − ADJDE


Additionally, postseason performance is encoded numerically (`POSTSEASON_NUM`) to enable analysis of how team metrics relate to tournament success.

---

## Algorithm Description

This project combines two main techniques:

### 1. Machine Learning Model

A classification model (Random Forest) is trained on historical data to estimate the probability that a team will win a game based on its statistical profile.

### 2. Monte Carlo Simulation

The tournament is simulated thousands of times. In each simulation:

- Matchups are played using predicted win probabilities  
- Winners advance through each round  
- A champion is determined  

By repeating this process many times, the model estimates:

- Championship probabilities  
- First-round upset likelihoods  
- Typical bracket progression  

---

## Tools Used

- **Python** – core programming language  
- **pandas** – data manipulation and preprocessing  
- **NumPy** – numerical computations  
- **scikit-learn** – machine learning model  
- **Streamlit** – interactive web application  
- **Altair** – visualization of probabilities  
- **matplotlib** – scatter plots and trend analysis  
- **joblib** – model and scaler persistence  

---

## Features

- Run customizable tournament simulations  
- View championship probability rankings  
- Identify likely first-round upsets (Seeds 11–16)  
- Explore relationships between team metrics and postseason success  
- Visualize full tournament brackets by region  

---

## Ethical Considerations

This model relies on historical data, which may introduce bias:

- Strong conferences may be overrepresented  
- Smaller or less prominent teams may be undervalued  
- Predictions reflect past performance, not real-time conditions (injuries, roster changes, etc.)

Additionally, this tool should not be used as a sole decision-making source for gambling or financial purposes. It is intended for educational and exploratory use.

To mitigate bias in future work, additional features such as player-level data, recent performance trends, or balanced datasets could be incorporated.

---

## Project Structure


March_Madness_Project/
│
├── app.py # Streamlit web application
├── simulation.py # Core simulation and game logic
├── models.py # Team class and probability calculations
├── test_simulation.py # Test cases for validation
├── cbb2_prepared.csv # Dataset
├── bracket_2025_round1.csv # Tournament structure
├── .gitignore # Ignored files
└── README.md # Project documentation


---

## How to Run

### 1. Install dependencies


pip install -r requirements.txt


### 2. Run the application


streamlit run app.py


### 3. Use the app

- Adjust the number of simulations using the slider  
- Click **Run Simulation**  
- View predictions, charts, and bracket results  

---

## Conclusion

This project demonstrates how machine learning and simulation techniques can be combined to model complex systems like tournament brackets. While predictions are not perfect, the application provides valuable insights into team performance, probability distributions, and uncertainty in competitive sports.