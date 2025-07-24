import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Cấu hình giao diện
st.set_page_config(page_title="Sales Forecast - ABC Manufacturing", layout="wide")
st.title("📊 Sales Forecasting with Linear Regression")

# 1. Load CSV trực tiếp từ GitHub
csv_url = "https://raw.githubusercontent.com/quangsang26/asmthayvu.github.io/refs/heads/main/abc_sales_data.csv"
try:
    df = pd.read_csv(csv_url)
    st.subheader("🔍 Data Preview")
    st.dataframe(df)

    # 2. Phân tích dữ liệu (EDA)
    st.subheader("📈 Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Weekly Sales Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Sales'], bins=20, kde=True, ax=ax2)
        st.pyplot(fig2)

    st.write("Scatter Plot: Advertising vs Sales")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=df['Advertising'], y=df['Sales'], ax=ax3)
    st.pyplot(fig3)

    st.write("Scatter Plot: Price vs Sales")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=df['Price'], y=df['Sales'], ax=ax4)
    st.pyplot(fig4)

    st.write("Sales by Week")
    fig5, ax5 = plt.subplots()
    sns.lineplot(x=df['Week'], y=df['Sales'], marker='o', ax=ax5)
    st.pyplot(fig5)

    # 3. Huấn luyện mô hình hồi quy tuyến tính
    st.subheader("🧠 Train Linear Regression Model")
    features = ['Advertising', 'Price', 'Online_Search']
    target = 'Sales'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

    # 4. Dự báo tùy chỉnh
    st.subheader("🔮 Predict Your Own Sales")

    adv = st.number_input("Advertising Spend", value=3000)
    price = st.number_input("Product Price", value=199.0)
    interest = st.number_input("Online Search Interest", value=500)

    input_data = pd.DataFrame([[adv, price, interest]], columns=features)
    predicted_sales = model.predict(input_data)[0]

    st.success(f"📦 Predicted Sales: **{predicted_sales:.2f} units**")

except Exception as e:
    st.error("❌ Failed to load the dataset. Please check the CSV URL or network connection.")
    st.code(str(e))
