# ==========================================
# E-COMMERCE CUSTOMER BEHAVIOUR ANALYSIS
# Using Only: Online_Sales.csv
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------
# 1️⃣ LOAD DATA
# ------------------------------------------
sales = pd.read_csv("Online_Sales.csv")

print("====== FIRST 5 ROWS ======")
print(sales.head())

print("\n====== DATA INFO ======")
print(sales.info())

# ------------------------------------------
# 2️⃣ HANDLE MISSING VALUES
# ------------------------------------------
print("\n====== MISSING VALUES BEFORE ======")
print(sales.isnull().sum())

# Fill numeric columns with mean
num_cols = sales.select_dtypes(include=np.number).columns
for col in num_cols:
    sales[col] = sales[col].fillna(sales[col].mean())

# Fill categorical columns with mode
cat_cols = sales.select_dtypes(include='object').columns
for col in cat_cols:
    sales[col] = sales[col].fillna(sales[col].mode()[0])

print("\n====== MISSING VALUES AFTER ======")
print(sales.isnull().sum())

# ------------------------------------------
# 3️⃣ REMOVE DUPLICATES
# ------------------------------------------
print("\nDuplicate rows:", sales.duplicated().sum())
sales.drop_duplicates(inplace=True)

# ------------------------------------------
# 4️⃣ DATA TYPE CONVERSION
# ------------------------------------------
if 'Transaction_Date' in sales.columns:
    sales['Transaction_Date'] = pd.to_datetime(sales['Transaction_Date'])

# ------------------------------------------
# 5️⃣ DESCRIPTIVE STATISTICS
# ------------------------------------------
print("\n====== DESCRIPTIVE STATISTICS ======")
print(sales.describe())

print("\nMean:")
print(sales.mean(numeric_only=True))

print("\nMedian:")
print(sales.median(numeric_only=True))

print("\nMode:")
print(sales.mode().iloc[0])

# ------------------------------------------
# 6️⃣ OUTLIER DETECTION (IQR METHOD)
# ------------------------------------------
if 'Quantity' in sales.columns:
    Q1 = sales['Quantity'].quantile(0.25)
    Q3 = sales['Quantity'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = sales[(sales['Quantity'] < lower) | (sales['Quantity'] > upper)]

    print("\nOutliers detected in Quantity:", outliers.shape[0])

# ------------------------------------------
# 7️⃣ FEATURE ENGINEERING
# ------------------------------------------
if 'Quantity' in sales.columns and 'Avg_Price' in sales.columns:
    sales['Total_Amount'] = sales['Quantity'] * sales['Avg_Price']

if 'Transaction_Date' in sales.columns:
    sales['Month'] = sales['Transaction_Date'].dt.month

# ------------------------------------------
# 8️⃣ VISUALIZATION
# ------------------------------------------

# Histogram
plt.figure()
sales['Quantity'].hist()
plt.title("Histogram of Quantity")
plt.show()

# Boxplot
plt.figure()
plt.boxplot(sales['Quantity'])
plt.title("Boxplot of Quantity")
plt.show()

# Scatter Plot
if 'Avg_Price' in sales.columns:
    plt.figure()
    plt.scatter(sales['Quantity'], sales['Avg_Price'])
    plt.xlabel("Quantity")
    plt.ylabel("Avg Price")
    plt.title("Quantity vs Avg Price")
    plt.show()

# Bar Chart
if 'Product_Category' in sales.columns:
    plt.figure()
    sales['Product_Category'].value_counts().plot(kind='bar')
    plt.title("Product Category Distribution")
    plt.show()

# ------------------------------------------
# 9️⃣ ENCODING
# ------------------------------------------
if 'Coupon_Status' in sales.columns:
    le = LabelEncoder()
    sales['Coupon_Status_Encoded'] = le.fit_transform(sales['Coupon_Status'])

# ------------------------------------------
# 🔟 SCALING
# ------------------------------------------
scaler = StandardScaler()
numeric_cols = sales.select_dtypes(include=np.number).columns
sales[numeric_cols] = scaler.fit_transform(sales[numeric_cols])

print("\nData Scaled Successfully")

# ------------------------------------------
# 1️⃣1️⃣ PCA
# ------------------------------------------
pca = PCA(n_components=2)
principal_components = pca.fit_transform(sales[numeric_cols])

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# ------------------------------------------
# 1️⃣2️⃣ SAVE CLEANED DATA
# ------------------------------------------
sales.to_csv("Processed_Online_Sales.csv", index=False)

print("\nANALYSIS COMPLETED SUCCESSFULLY ")
