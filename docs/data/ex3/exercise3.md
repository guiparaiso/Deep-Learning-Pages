
# Exercise 3 — Preparing Real-World Data for a Neural Network

## 1. Describe the Data

### Objective
The **Spaceship Titanic** dataset simulates passenger records from a spaceship that suffered an accident.  
The goal is to **predict whether each passenger was transported to another dimension**, represented by the column:

```

Transported → Boolean (True/False)

````

- **True** → Passenger was transported.  
- **False** → Passenger was not transported.  

This is a **binary classification** problem.

---

### Features

| Type | Feature | Description |
|------|----------|-------------|
| **Categorical** | HomePlanet | Passenger’s home planet (Earth, Europa, Mars) |
| **Categorical** | CryoSleep | Whether the passenger was in cryosleep during the voyage |
| **Categorical** | Cabin | Cabin identifier (deck/side/number) |
| **Categorical** | Destination | Destination planet |
| **Numerical** | Age | Passenger’s age |
| **Numerical** | RoomService, FoodCourt, ShoppingMall, Spa, VRDeck | Amount spent in each onboard amenity |
| **Categorical** | Name | Passenger name (often not useful for modeling) |
| **Categorical/Numerical** | PassengerId | Unique passenger identifier |
| **Target** | Transported | Whether the passenger was transported (True/False) |

---

### Missing Values Investigation

Example code:
```python
import pandas as pd
df = pd.read_csv("spaceship_titanic.csv")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])
````

Typical results (approximate):

| Column       | Missing Values |
| ------------ | -------------- |
| HomePlanet   | ~200           |
| CryoSleep    | ~200           |
| Cabin        | ~200           |
| Destination  | ~180           |
| Age          | ~180           |
| RoomService  | ~180           |
| FoodCourt    | ~200           |
| ShoppingMall | ~210           |
| Spa          | ~200           |
| VRDeck       | ~190           |
| Name         | ~100           |
| Transported  | 0              |

Several columns contain missing data, both numerical and categorical.

---

## 2. Preprocess the Data

### a) Handle Missing Data

**Strategy:**

* **Numerical columns:** Replace missing values with the **median** (less sensitive to outliers).
* **Categorical columns:** Replace missing values with the **mode** (most frequent value) or `"Unknown"`.

```python
num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
cat_cols = ['HomePlanet','CryoSleep','Cabin','Destination']

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
```

**Justification:**
Using median/mode preserves all rows and avoids information loss — essential for stable neural network training.

---

### b) Encode Categorical Features

Convert non-numeric features into numeric form using **One-Hot Encoding**:

```python
df = pd.get_dummies(df, columns=['HomePlanet','CryoSleep','Destination'], drop_first=True)
```

Example output columns:

```
HomePlanet_Europa, HomePlanet_Mars, CryoSleep_True, Destination_TRAPPIST-1e
```

**Why:** Neural networks require numeric inputs; one-hot encoding avoids false ordinal relationships.

---

### c) Normalize / Standardize Numerical Features

Since the neural network uses **tanh** activation (range = [-1, 1]), inputs must be centered near zero.
Use **Standardization** (mean = 0, std = 1) for stable gradients:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
```

**Why:**
`tanh` saturates for large values; scaling keeps inputs in its sensitive region, improving learning speed and convergence.

---

## 3. Visualize the Results

### Example: Histogram Before and After Scaling

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10,4))
df_raw = pd.read_csv("spaceship_titanic.csv")

ax[0].hist(df_raw['Age'].dropna(), bins=30, edgecolor='k')
ax[0].set_title("Age before scaling")

ax[1].hist(df['Age'], bins=30, edgecolor='k')
ax[1].set_title("Age after scaling (Standardized)")

plt.show()
```

<p align="center">
  <img src="../data/ex3/hist_age_before.png" alt="Age before scaling" width="400">
  <img src="../data/ex3/hist_age_after.png" alt="Age after scaling" width="400">
</p>

**Observation:**

* Before scaling: values between 0–80, centered near 30.
* After scaling: mean ≈ 0, std ≈ 1 (range roughly -2 to +2).
  This suits the `tanh` activation function’s centered output range.

---

## 4. Summary

**Objective:**
Predict whether a passenger was transported (`Transported`).

**Steps Taken:**

1. Filled missing values (median/mode).
2. One-Hot Encoded categorical variables.
3. Standardized numeric columns for `tanh` activation.
4. Visualized transformations via histograms.

**Why it matters:**
Proper preprocessing ensures clean, scaled, balanced inputs — key for stable neural network training and better accuracy.

---

