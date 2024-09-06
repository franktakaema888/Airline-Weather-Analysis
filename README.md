# Flight Delay Prediction and Classification Using Weather Data

## Project Overview

This project aims to predict flight delays based on weather conditions using machine learning models. The key motivation behind the project is to fill the gap in studies that focus on the relationship between a broader range of weather parameters and their impact on flight delays. The project explores various deep learning models, including CNN, LSTM, and an ensemble model (stacking), to enhance the accuracy of predictions and provide insights into the influence of weather on airline travel time.

## Project Objectives

1. **Analyze the Impact of Weather on Air Travel**  
   Examine how different weather conditions affect airline arrival times and identify key weather parameters that influence delays.

2. **Utilize Deep Learning**  
   Apply deep learning models such as CNNs and LSTMs to better understand the effects of a broader range of weather conditions on flight travel times.

3. **Improve Safety and Efficiency**  
   Contribute to improved safety measures and operational efficiency for airlines by providing comprehensive insights that consider a wide range of weather variables.

## Models Explored

- **Convolutional Neural Network (CNN):**  
   A baseline and improved CNN model were used to capture temporal patterns in the data related to weather and flight delays.

- **Long Short-Term Memory (LSTM):**  
   LSTM models were used to capture long-term dependencies in the sequential weather data.

- **Ensemble Model (Stacking):**  
   A combination of CNN and LSTM models were stacked together to form an ensemble learning model. This model aimed to leverage the strengths of both CNN and LSTM, but faced performance challenges.

## Final Model: Improved CNN

The Improved CNN model demonstrated better performance in predicting both "Delayed" and "Not Delayed" flights. It achieved an accuracy of approximately **77%**, with a strong balance between precision and recall for delayed flights.

## Key Results

- **Improved CNN Model:**
  - Accuracy: ~77%
  - Precision for "Delayed" class: Significantly improved over the baseline.
  - Balanced predictions for both "Delayed" and "Not Delayed" classes.

- **Ensemble (Stacking) Model:** 
  - Accuracy: 50% (equivalent to random guessing).
  - Failed to predict the "Delayed" class effectively.

## Dataset

- **Source:** The dataset includes weather and flight data. It contains weather features such as temperature, wind speed, and precipitation, as well as flight features like actual elapsed time and arrival delay.

- **Features:** The dataset contains a mixture of categorical and numerical variables, including engineered features to capture weather and flight time interaction terms.

## How to Run the Project

1. Clone the repository.
    ```bash
    git clone <repository-url>
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook or Python script to train and evaluate the models:
    ```bash
    python train_model.py
    ```
4. Explore the results through the classification report and confusion matrices.

## Requirements

- Python 3.x
- TensorFlow/Keras
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn

## File Structure

```
├── data/                   # Contains the dataset
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks
├── scripts/                # Python scripts for training and evaluation
├── results/                # Model results and visualizations
└── README.md               # Project description
```

## Future Work

- **Model Optimization:** Further improvement of the ensemble model to achieve better performance.
- **Feature Importance:** Investigating feature importance to understand which weather variables have the most impact on delays.
- **Real-Time Predictions:** Incorporate real-time weather data for live predictions of flight delays.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README file gives a structured overview of your project, providing necessary details to get started and understand the goals, models, results, and how to run the code.
