# Calculus for AI/ML

*Calculus helps us understand how things change. In machine learning, it is mainly used for optimization — that is, improving model performance by reducing error.*

- If linear algebra is about structure and statistics is about data, calculus is about **change and optimization**.

- In machine learning:
  - we use calculus to update model parameters
  - we use derivatives to minimize error
  - we use gradients to learn from data

---

## What is Calculus?

- Calculus studies how a function changes.

- There are two main parts:
  - Differentiation → rate of change  
  - Integration → accumulation  

- In ML, we mostly use **differentiation**.

---

## 1. Function

- A function is a relationship between input and output.

Example:

y = x²

- If x changes, y also changes.

---

## 2. Derivative

- Derivative tells how fast a function changes.

- It is also called the **slope**.

Example:

y = x²  
Derivative = 2x  

- At x = 2 → slope = 4  

---

## 3. Slope Intuition

- Slope tells direction:
  - Positive slope → increasing  
  - Negative slope → decreasing  
  - Zero slope → flat (minimum or maximum)  

---

## 4. Partial Derivative

- When function has multiple variables, we take derivative w.r.t one variable.

Example:

z = x² + y²  

- Partial derivative w.r.t x = 2x  
- Partial derivative w.r.t y = 2y  

---

## 5. Gradient

- Gradient is a vector of all partial derivatives.

Example:

z = x² + y²  

Gradient = [2x, 2y]

- Gradient shows direction of maximum increase.

---

## 6. Cost Function

- Cost function measures error in prediction.

Example:

Error = (actual - predicted)²  

- Goal: minimize this error

---

## 7. Gradient Descent

- Gradient descent is used to minimize error.

Steps:
1. Start with random values  
2. Calculate gradient  
3. Update values  
4. Repeat  

- This is how models learn.

---

## 8. Learning Rate

- Learning rate controls step size.

- If too large → overshoot  
- If too small → slow learning  

---

## 9. Minima

- Local minimum → small dip  
- Global minimum → lowest point  

Goal:
Find global minimum

---

## 10. Calculus in Machine Learning

- Used in:
  - Linear regression  
  - Neural networks  
  - Deep learning  
  - Optimization algorithms  

---

## Coding Tasks (use Colab or notebook)

### 1. Basic Function

```python
import numpy as np

x = np.array([1, 2, 3, 4])
y = x**2

print("x:", x)
print("y = x^2:", y)
