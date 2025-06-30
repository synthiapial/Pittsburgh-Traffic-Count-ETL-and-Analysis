# Author: Synthia Pial
# Project: Pittsburgh Traffic Count Analysis with Visualization and Prediction

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1. Extract - Load CSV
def extract_data(csv_file):
    print("Extracting data...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows.")
    print(f"Columns: {list(df.columns)}")
    return df


# Step 2. Transform - Clean and aggregate
def transform_data(df):
    print("Transforming data...")

    # Drop rows with missing key data
    df = df.dropna(subset=['neighborhood', 'average_daily_car_traffic', 'count_start_date'])

    # Parse dates
    df['count_start_date'] = pd.to_datetime(df['count_start_date'], errors='coerce')
    df = df.dropna(subset=['count_start_date'])

    # Convert traffic to numeric
    df['average_daily_car_traffic'] = pd.to_numeric(df['average_daily_car_traffic'], errors='coerce')
    df = df.dropna(subset=['average_daily_car_traffic'])

    print(f"Cleaned to {len(df)} rows.")
    return df


# Step 3. Load - Save to SQLite
def load_to_sqlite(df, db_name, table_name):
    print(f"Loading to database: {db_name}")
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print("Data saved to SQLite.")


# Step 4. Visualization - Top neighborhoods and trends
def visualize_data(df):
    print("Visualizing data...")

    # Top 10 neighborhoods by avg traffic
    top_n = (
        df.groupby('neighborhood')['average_daily_car_traffic']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,6))
    top_n.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Neighborhoods by Average Daily Car Traffic')
    plt.ylabel('Average Daily Car Traffic')
    plt.xlabel('Neighborhood')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top_neighborhoods.png')
    plt.show()

    # Monthly trend
    df['month'] = df['count_start_date'].dt.to_period('M')
    monthly_trend = df.groupby('month')['average_daily_car_traffic'].mean()

    plt.figure(figsize=(12,6))
    monthly_trend.plot()
    plt.title('Monthly Average Daily Car Traffic')
    plt.ylabel('Avg Daily Car Traffic')
    plt.xlabel('Month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_trend.png')
    plt.show()


# Step 5. Simple Prediction - Linear regression over time
def predict_traffic(df):
    print("Running prediction model...")

    # Aggregate by day
    daily_data = (
        df.groupby('count_start_date')['average_daily_car_traffic']
        .mean()
        .reset_index()
        .sort_values('count_start_date')
    )

    # Prepare features
    daily_data['ordinal_date'] = daily_data['count_start_date'].map(pd.Timestamp.toordinal)
    X = daily_data[['ordinal_date']]
    y = daily_data['average_daily_car_traffic']

    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    print(f"Model coefficient: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

    # Predict future dates
    future_dates = pd.date_range(daily_data['count_start_date'].max(), periods=30, freq='D')
    future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1,1)
    predictions = model.predict(future_ordinals)

    # Plot results
    plt.figure(figsize=(12,6))
    plt.scatter(daily_data['count_start_date'], y, label='Actual', alpha=0.5)
    plt.plot(future_dates, predictions, color='red', label='Predicted Trend')
    plt.title('Traffic Volume Prediction')
    plt.xlabel('Date')
    plt.ylabel('Average Daily Car Traffic')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_trend.png')
    plt.show()

    print("Prediction complete!")


# Step 6. Main
def main():
    csv_file = 'pittsburgh_traffic_counts_cleaned.csv'
    db_name = 'traffic_analysis.db'
    table_name = 'TrafficData'

    # ETL steps
    raw_data = extract_data(csv_file)
    cleaned_data = transform_data(raw_data)
    load_to_sqlite(cleaned_data, db_name, table_name)

    # Analysis
    visualize_data(cleaned_data)
    predict_traffic(cleaned_data)

    print("Analysis pipeline complete!")


if __name__ == "__main__":
    main()


