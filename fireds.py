#Analysis of Forest Fire Intensity Using Satellite Data

#Forest fires are increasing globally and pose serious environmental risks.
#This project aims to analyze satellite-based fire data and predict fire
#intensity using machine learning techniques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#Load Data
df = pd.read_csv(r"C:\Users\ACER\Downloads\modis_2024_all_countries\modis\2024\modis_2024_United_States.csv")

print(df.head())
print(df.info())
print(df.describe())


# Adding Style

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# Logical color theme
main_color = "#4CAF50"      # green
secondary_color = "#FF9800" # orange
accent_color = "#F94144"    # red
soft_blue = "#6FA8DC"

pastel_colors = ["#90BE6D", "#F9C74F", "#F8961E", "#F94144"]

#Rename Columns
df = df.rename(columns={
    'brightness': 'Fire Temperature',
    'frp': 'Fire Intensity',
    'bright_t31': 'Background Temperature',
    'confidence': 'Detection Confidence'
})

#Data Cleaning
df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
df = df.dropna()

numeric_df = df.select_dtypes(include=np.number)


# HISTOGRAMS


# Fire Temperature
plt.figure()

sns.histplot(
    df['Fire Temperature'],
    bins=50,
    kde=True,
    color="#A8E6CF",        
    edgecolor="white",
    alpha=0.7,              
    line_kws={"color": "#2E7D32", "linewidth": 2}  
)

plt.title("Distribution of Fire Temperature")
plt.xlabel("Fire Temperature")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Fire Intensity 
threshold = df['Fire Intensity'].quantile(0.99)
filtered_df = df[df['Fire Intensity'] <= threshold]

plt.figure()
sns.histplot(
    filtered_df['Fire Intensity'],
    bins=50,
    kde=True,
    color=secondary_color
)
plt.title("Distribution of Fire Intensity")
plt.xlabel("Fire Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# BOXPLOTS

#For Fire Temperature
plt.figure()
sns.boxplot(x=df['Fire Temperature'], color="#90BE6D")
plt.title("Fire Temperature Boxplot")
plt.tight_layout()
plt.show()

#For Fire Intensity
plt.figure()
sns.boxplot(x=df['Fire Intensity'], color="#F8961E")
plt.title("Fire Intensity Boxplot")
plt.tight_layout()
plt.show()


# SCATTER PLOT

plt.figure()
sns.regplot(
    x=df['Fire Temperature'],
    y=df['Fire Intensity'],
    scatter_kws={'color': soft_blue, 'alpha':0.5},
    line_kws={'color': accent_color}
)
plt.xlabel("Fire Temperature")
plt.ylabel("Fire Intensity")
plt.title("Fire Temperature vs Fire Intensity")
plt.tight_layout()
plt.show()


# HEATMAP

plt.figure(figsize=(10,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# BAR CHARTS

# Satellite
sat_counts = df['satellite'].value_counts()
bars = plt.bar(sat_counts.index, sat_counts.values, color=pastel_colors)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, yval, int(yval),
             ha='center', va='bottom')

plt.title("Satellite Distribution")
plt.xlabel("Satellite")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Day vs Night
dn_counts = df['daynight'].value_counts()
bars = plt.bar(dn_counts.index, dn_counts.values, color=pastel_colors)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, yval, int(yval),
             ha='center', va='bottom')

plt.title("Day vs Night Fires")
plt.xlabel("Day/Night")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Monthly trend
df['month'] = df['acq_date'].dt.month
month_counts = df['month'].value_counts().sort_index()

plt.figure()
month_counts.plot(kind='bar', color=main_color)
plt.title("Fire Occurrence by Month")
plt.xlabel("Month")
plt.ylabel("Number of Fires")
plt.tight_layout()
plt.show()


# DONUT CHART
#For satellite contribution
plt.figure()

wedges, texts, autotexts = plt.pie(
    sat_counts,
    labels=sat_counts.index,
    autopct='%1.1f%%',
    colors=pastel_colors,
    startangle=90,
    wedgeprops={'edgecolor': 'black', 'linewidth': 1}
)

# Inner circle with border
centre_circle = plt.Circle(
    (0, 0),
    0.70,
    fc='white',
    edgecolor='black',
    linewidth=1.5
)
plt.gca().add_artist(centre_circle)

plt.title("Satellite Contribution (Donut Chart)")
plt.show()


# LINEAR REGRESSION

X = df[['Fire Temperature', 'Detection Confidence', 'Background Temperature']]
y = df['Fire Intensity']

#Model Training

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# MODEL EVALUATION

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# ACTUAL VS PREDICTED Fire

plt.figure()
plt.scatter(y_test, y_pred, color=main_color, alpha=0.6)
plt.xlabel("Actual Fire Intensity")
plt.ylabel("Predicted Fire Intensity")
plt.title("Actual vs Predicted Fire Intensity")
plt.tight_layout()
plt.show()
