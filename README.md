# Titanic Data Preprocessing - README

This document outlines the step-by-step data preprocessing workflow applied to the Titanic dataset using only **NumPy**, **Pandas**, and **Matplotlib**. It also includes explanations for questions asked throughout the process.

---

#### Step 1: Import Dataset and Explore Basic Info

* Loaded the Titanic dataset using `pd.read_csv()`.
* Used `df.info()`, `df.describe()`, `df.isnull().sum()` to check for:

  * Column data types
  * Missing values
  * Basic statistics

---

## 🧹 Step 2: Handle Missing Values

* Checked columns with missing values:

  * `Age` had many missing values
  * `Cabin` had too many missing values (optional to drop)
  * `Embarked` had few missing values
* Imputation Strategy:

  * `Age`: Filled using **median**

    ```python
    df['Age'].fillna(df['Age'].median(), inplace=True)
    ```
  * `Embarked`: Filled with **mode** (most frequent)

    ```python
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    ```

---

## 🔠 Step 3: Encode Categorical Variables

* Converted `Sex` to numerical using mapping:

  ```python
  df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
  ```
* `Embarked` encoded with:

  ```python
  df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
  ```
* For `Ticket`, encoded like this:

  ```python
  df['Ticket'] = df['Ticket'].apply(lambda x: 1 if x.isdigit() else 0)
  ```
* Note: Name and Cabin are complex and were not used for modeling yet.

### What is "Embarked"?

* It indicates **boarding port**:

  * `S` = Southampton (mostly 3rd class)
  * `Q` = Queenstown (2nd class)
  * `C` = Cherbourg (1st class)

###  Encoding `Ticket` and `Name`

* Tickets with digits = likely 3rd class
* Tickets with letters = higher classes
* Simplified ticket encoding by checking if it's numeric

---

## 📏 Step 4: Normalize / Standardize Numerical Features

### 🔄 Standardization:

* Used Z-score formula:

  ```python
  df[col] = (df[col] - df[col].mean()) / df[col].std()
  ```
* Applied to columns like `Fare`, `Age`, `SibSp`, `Parch`, `Pclass`
* Excluded `Name`, `Ticket`, `Cabin`, `Embarked`, `PassengerId`

### Difference Between Standardization and Normalization:

| Feature  | Standardization                      | Normalization (Min-Max) |
| -------- | ------------------------------------ | ----------------------- |
| Formula  | (x - mean) / std                     | (x - min) / (max - min) |
| Range    | Centered around 0                    | 0 to 1                  |
| Use Case | Most ML models, normal distributions | KNN, Deep Learning      |

---

##  Step 5: Outlier Detection & Removal

###  Visualized Outliers using Boxplots:

```python
plt.boxplot(df['Fare'])
plt.boxplot(df['Age'])
```

* Boxplot components:

  * Box = middle 50%
  * Line = median
  * Dots = outliers

### ❓ What is an Outlier?

* Values beyond the range:

  ```
  Q1 - 1.5 * IQR  OR  Q3 + 1.5 * IQR
  ```

### 🧹 Removed Outliers from `Fare` and `Age`:

```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
print("After removing outliers from Fare:", df_cleaned.shape)
```

### 📉 Result:

* Original rows: 891
* After removing outliers: 601

---

## ✅ Summary of What Happened:

* Missing values were filled (Age, Embarked)
* Categorical values were encoded (Sex, Embarked, Ticket)
* Numerical features were standardized
* Outliers were visualized and removed using boxplot + IQR

---

This README serves as a complete walkthrough of Titanic data cleaning steps and explanations using basic data science tools.
