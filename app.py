from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import os

app = Flask(__name__)

# ================= LOAD DATA =================

data = pd.read_csv("advanced_dataset_with_total.csv")

# ================= CLEAN DATA =================
data['Date'] = pd.to_datetime(data['Date'])
data['State'] = data['State'].astype(str).str.strip().str.title()
data['Disaster_Type'] = data['Disaster_Type'].astype(str).str.strip().str.title()
data['Severity_Level'] = data['Severity_Level'].astype(str).str.strip().str.title()

@app.route('/', methods=['GET', 'POST'])
def home():

    disaster = request.form.get('disaster')
    state = request.form.get('state')
    severity = request.form.get('severity')

    df = data.copy()

    # ================= FILTER =================
    if disaster and disaster != "All":
        df = df[df['Disaster_Type'] == disaster]

    if state and state != "All":
        df = df[df['State'] == state]

    if severity and severity != "All":
        df = df[df['Severity_Level'] == severity]

    df = df.sort_values('Date')

    # HANDLE EMPTY
    if df.empty:
        df = data.copy()

    # ================= KPI =================
    total_disasters = df['Total_Disaster'].sum()
    total_affected = int(df['Affected_People'].sum())
    avg_response = round(df['Response_Time_hr'].mean(), 2)
    high_risk = len(df[df['Severity_Level'] == 'High'])

    # ================= ALERT =================
    alert = ""
    if high_risk > 5:
        alert = "⚠️ High Risk Alert! Immediate action required"

    os.makedirs("static", exist_ok=True)

    # ================= PIE =================
    plt.figure()
    df['Severity_Level'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Severity Distribution")
    plt.ylabel('')
    plt.savefig("static/pie.png")
    plt.close()

    # ================= BAR =================
    plt.figure()
    df['State'].value_counts().plot(kind='bar')
    plt.title("State-wise Disasters")
    plt.xticks(rotation=45)
    plt.savefig("static/bar.png")
    plt.close()

    # ================= MAP =================
    df['Severity_Level'] = df['Severity_Level'].fillna('Low')

    colors = df['Severity_Level'].map({
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }).fillna('blue')

    plt.figure()
    plt.scatter(df['Longitude'], df['Latitude'], c=colors)
    plt.title("Disaster Locations (Severity Based)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("static/map.png")
    plt.close()

    # ================= PREDICTION (Affected People) =================
    daily = df.groupby('Date')['Affected_People'].sum().reset_index()

    daily['Days'] = (daily['Date'] - daily['Date'].min()).dt.days

    X = daily[['Days']]
    y = daily['Affected_People']

    # 🔥 Polynomial Regression
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # 🔥 Next 1 Year
    future_days = np.arange(daily['Days'].max()+1, daily['Days'].max()+366).reshape(-1,1)
    future_dates = pd.date_range(start=daily['Date'].max()+pd.Timedelta(days=1), periods=365)

    future_poly = poly.transform(future_days)
    pred = model.predict(future_poly)

    # TOTAL FUTURE AFFECTED PEOPLE
    total_future = int(sum(pred))

    # GRAPH
    plt.figure(figsize=(10,5))
    plt.plot(daily['Date'], y, label="Actual Affected People")
    plt.plot(future_dates, pred, 'r--', label="Predicted Affected People")

    plt.xlabel("Date")
    plt.ylabel("Affected People")
    plt.title("Future Affected People Prediction (Next 1 Year)")
    plt.legend()
    plt.grid()

    plt.savefig("static/output.png")
    plt.close()

    # ================= DROPDOWN =================
    disasters = list(data['Disaster_Type'].dropna().unique())
    states = list(data['State'].dropna().unique())
    severities = list(data['Severity_Level'].dropna().unique())

    return render_template("index.html",
                           disasters=disasters,
                           states=states,
                           severities=severities,
                           total=total_disasters,
                           affected=total_affected,
                           response=avg_response,
                           high=high_risk,
                           alert=alert,
                           future=total_future)

# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True, port=5001)