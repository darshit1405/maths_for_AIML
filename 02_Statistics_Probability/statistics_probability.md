# Statistics & Probability

*Statistics and probability help us understand data, uncertainty, and patterns. This file covers mean, variance, probability basics, correlation, distributions, and their role in machine learning.*

- Statistics helps us summarize data.
- Probability helps us deal with uncertainty.

- In machine learning:
  - we use statistics to understand data
  - we use probability to make predictions

- If linear algebra is about structure, statistics is about **understanding data behavior**.

---

## Mean (Average)

- Mean is the average of all values.

Example:
[1, 2, 3, 4, 5]

Mean = (1+2+3+4+5) / 5 = 3

- Mean is useful to understand the central value of data.

---

## Median

- Median is the middle value after sorting.

Example:
[1, 2, 3, 4, 5] → Median = 3

- Median is useful when data has outliers.

---

## Variance

- Variance measures how spread out the data is.

- If variance is small → values are close to mean  
- If variance is large → values are far from mean  

---

## Standard Deviation

- Standard deviation is the square root of variance.

- It tells how much values deviate from the mean.

---

## Probability Basics

- Probability tells how likely something is.

Formula:
Probability = favorable outcomes / total outcomes

Example:
If 3 out of 10 outcomes are favorable:

Probability = 3 / 10 = 0.3

---

## Conditional Probability

- Conditional probability means probability of one event given another.

Example:
Probability of rain given clouds

---

## Covariance

- Covariance shows how two variables change together.

- Positive → increase together  
- Negative → move opposite  

---

## Correlation

- Correlation measures strength of relationship.

Values:
- +1 → strong positive  
- 0 → no relation  
- -1 → strong negative  

---

## Normal Distribution

- Normal distribution is a bell-shaped curve.

- Most values are near the mean  
- Few values are far from the mean  

---

## Sampling

- Sampling means selecting a small portion of data.

- Used when dataset is large.

---

## Statistics in Machine Learning

- Mean → data normalization  
- Variance → model evaluation  
- Probability → classification  
- Correlation → feature selection  

Example:
If two features are highly correlated, one can be removed.

---

## Coding Tasks (use Colab or notebook)

### 1. Mean, Median, Variance, Standard Deviation

```python
import numpy as np

data = np.array([1, 2, 3, 4, 5])

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))

---

### 2. Probability

```python
favorable = 3
total = 10

print("Probability:", favorable / total)

---

### 3. Conditional Probability

```python
p_a_and_b = 0.2
p_b = 0.5

print("P(A|B):", p_a_and_b / p_b)

---

### 4. Covariance and Correlation

```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])

print("Covariance:\n", np.cov(x, y))
print("Correlation:\n", np.corrcoef(x, y))

---

### 5. Normal Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(0, 1, 1000)

plt.hist(data, bins=30)
plt.title("Normal Distribution")
plt.show()

---

### 6. Sampling

```python
import numpy as np

data = np.array([1,2,3,4,5,6,7,8,9,10])

sample = np.random.choice(data, size=3)

print("Sample:", sample)
