# Dogecoin (DOGE) Price Prediction with Machine Learning

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)
[](https://www.tensorflow.org/)

A machine learning project that aims to predict the closing price of Dogecoin (DOGE-USD) using historical data and a Long Short-Term Memory (LSTM) neural network model.

-----

### ⚠️ High-Risk Warning (Disclaimer)

**This project was created ONLY for educational and technological exploration purposes.** The cryptocurrency market, especially assets like Dogecoin, is extremely **volatile and unpredictable**. Its price is heavily influenced by market sentiment, news, social media hype, and other external factors that cannot be fully captured by this model.

**NEVER USE PREDICTIONS FROM THIS PROJECT AS A BASIS FOR INVESTMENT DECISIONS.** Investing in cryptocurrencies carries a very high risk. The author is not responsible for any financial losses you may experience.

-----

### 📝 Project Description

This project explores the application of deep learning for time-series forecasting on Dogecoin's price. An LSTM model is built to learn long-term patterns and dependencies from historical price data, with the goal of predicting the closing price (`Close Price`) for the next day. This project covers the entire workflow, from data acquisition, preprocessing, model design, and training, to results evaluation.

### 🎯 Background

Dogecoin, which started as a meme, has become one of the most well-known cryptocurrencies in the world, with extreme price fluctuations. Its highly speculative nature makes it an interesting case study for testing the limits of predictive models. This project attempts to answer the question: "How well can an LSTM model capture patterns amidst the 'noise' and volatility of the Dogecoin market?"

### ✨ Key Features

  - **Dynamic Data Acquisition**: Uses the `yfinance` library to fetch the latest Dogecoin (`DOGE-USD`) price data from Yahoo Finance.
  - **Time-Series Data Analysis**: Visualization of trends, volatility, and historical patterns of Dogecoin's price.
  - **LSTM Model for Volatility**: Implements a Long Short-Term Memory (LSTM) network designed for sequential data.
  - **Visual Evaluation**: Presents a graphical comparison between predicted and actual prices to visually assess the model's performance.

### 📊 Dataset

The data used is the historical price data for **Dogecoin (DOGE-USD)**, which is fetched dynamically. The main features in the dataset include:

  - `Open`: The opening price
  - `High`: The highest price in a day
  - `Low`: The lowest price in a day
  - `Close`: **(Target Feature)** The closing price
  - `Volume`: The trading volume

### 🛠️ Tech Stack

  - **Language**: Python 3.8+
  - **Data Fetching**: `yfinance`
  - **Analysis & Computation**: Pandas, NumPy
  - **Visualization**: Matplotlib, Plotly
  - **Deep Learning**: TensorFlow (Keras API)
  - **Preprocessing**: Scikit-learn

### 🚀 Installation and Usage

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment (highly recommended):**

    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate | macOS/Linux: source venv/bin/activate
    ```

3.  **Install all required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook to see the entire process:**

    ```bash
    jupyter notebook notebooks/dogecoin_price_prediction.ipynb
    ```

### 📁 Project Structure

```
.
├── notebooks/
│   └── dogecoin_price_prediction.ipynb  # Main notebook for analysis and modeling
├── models/
│   └── doge_lstm_model.h5               # The trained LSTM model file
├── requirements.txt                     # List of Python dependencies
└── README.md                            # This file
```

### 🧠 Methodology and Model

This model uses a **Long Short-Term Memory (LSTM)** architecture, which is ideal for time-series data due to its ability to remember information over long periods.

**Model Workflow:**

1.  **Data Acquisition**: Download historical `DOGE-USD` data.
2.  **Preprocessing**:
      - Focus on the `Close` column as the main feature for prediction.
      - Scale the data to a range of [0, 1] using `MinMaxScaler` to stabilize the training process.
3.  **Sequence Creation**: Transform the data into a sequential format. For example, use data from the last 60 days (`time_step = 60`) to predict the price on the 61st day.
4.  **Model Architecture**: Build an LSTM model with several layers, `Dropout` layers to reduce overfitting, and a final `Dense` layer as the output.
5.  **Training**: Train the model on the training data using the `Adam` optimizer and `mean_squared_error` as the loss function.
6.  **Evaluation**: Evaluate the model's performance on the test data using the **Root Mean Squared Error (RMSE)** metric.

### 🤝 Contributing

Feel challenged to improve this model? Contributions are very welcome\! Please **Fork** this repository, create a new **Branch**, make your changes, and submit a **Pull Request**.

### 📄 License

This project is licensed under the **MIT License**.
