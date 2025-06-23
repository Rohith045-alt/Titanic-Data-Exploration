
#STEP 1: (Import the dataset and explore basic info)

import pandas as pd
import seaborn as sns
df = pd.read_csv('Titanic-Dataset.csv')
df.head()
df.shape
df.info()

#STEP 2: (Handle missing values using mean/median/imputation)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')

#STEP 3: (Convert categorical features into numerical using encoding)

print(df.select_dtypes(include=['object']).columns) #Index(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Cabin'] = df['Cabin'].apply(lambda x: 1 if x != 'Unknown' else 0)
df['Ticket'] = df['Ticket'].apply(lambda x: 1 if x.isdigit() else 0)

#STEP 4: (Normalize/standardize the numerical features using 'Fare' and 'Age')

num_cols = df.select_dtypes(include='number').columns
print(num_cols)  #Index(['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], dtype='object')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].hist(df['Fare'].dropna(), bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Original Fare Distribution')

axes[0, 1].hist(df['Age'].dropna(), bins=30, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Original Age Distribution')

df['Fare_standardized'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()
df['Age_standardized'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

axes[1, 0].hist(df['Fare_standardized'].dropna(), bins=30, color='dodgerblue', edgecolor='black')
axes[1, 0].set_title('Standardized Fare Distribution')

axes[1, 1].hist(df['Age_standardized'].dropna(), bins=30, color='mediumseagreen', edgecolor='black')
axes[1, 1].set_title('Standardized Age Distribution')

plt.tight_layout()
plt.show()

#STEP 5: (Visualize outliers using boxplots and remove them using 'Fare' and 'Age')
#visulization
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot(df['Fare'].dropna(), patch_artist=True)
plt.title('Boxplot of Fare')
plt.subplot(1, 2, 2)
plt.boxplot(df['Age'].dropna(), patch_artist=True)
plt.title('Boxplot of Age')
plt.tight_layout()
plt.show()

#remvoving
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
print("After removing outliers from Fare:", df_cleaned.shape)
