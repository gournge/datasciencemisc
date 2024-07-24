# Quant interview

## 0xfdf x.com thread

Since my knowledge is limited, in my solutions I used Wikipedia, Rudin's Princples of Mathematical Analysis, Axler's Linear Algebra Done Right and James' An Introduction to Statistical Learning. 

Here is the [link to the X.com thread](https://x.com/0xfdf/status/1815166904010506620). 

The questions:

Question 1: I tell you that I am regressing $Y$ on $X$. Describe what I probably mean as rigorously and completely as you can, without sacrificing generality.

[Answer](xfdf-answers/q1.md)

Question 2: I have a univariate linear model of $X$ and $Y$, both $n \times 1$ column vectors. Describe what happens when I try to estimate this model with an exact copy of $X$ added as a second exogenous variable, and why, as rigorously and completely as you can.

[Answer](xfdf-answers/q2.md)

</details>
<hr>
<details>
  <summary>Question 3: Suppose I estimated my initial univariate linear model with a Ridge penalty. Describe what this means in the language of norms. Be very specific and precise.</summary>
</details>
<details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 4: Now, what happens to the estimated coefficients of my linear model if I estimate it with a Ridge penalty after adding the second exogenous column that is identical to $X$?</summary>
<details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 5: Tell me that answer if I instead estimate the model with the copied data under a Lasso penalty. Describe the lasso penalty in the language of norms.</summary>
</details>
</details>
<details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 7: On this whiteboard, describe how to alter the design matrix of the linear model so that it has a Ridge penalty of $\lambda$.</summary>
</details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 8: Describe the difference between iterated regression and matrix decomposition (rotation) to transform an $N \times M$ matrix so that each of the $N$ columns is pairwise orthogonal.</summary>
<details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 9: You have an $N \times M$ matrix, with $N \ll M$. Let's say $M$ is 2000 and $N$ is 200. Describe what happens if you try to estimate the covariance matrix, in terms of both numerical stability and error.</summary>
<details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 10: You ascertain z-scores that effectively project the $M$ columns to a 20-day space. Your scores are known, an $N \times 20$ matrix. Your $Y$ is known, an $N \times 1$ column vector. Estimate the reduced column space of $X'_1, \ldots, X'_{20}$.</summary>
<details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 11: Walk me through the estimation of principal components of an $N \times M$ matrix whose columns are variables of interest. Be specific in terms of eigenvalues and eigenvectors.</summary>
<details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 12: What algorithm would you use, and why, to estimate the covariance matrix of an $N \times M$ matrix?</summary>
</details>
  <summary>My solution</summary>
</details>
<hr>
  <summary>Question 13: I hand you a covariance matrix. Tell me how you would quickly ascertain if its underlying matrix columns are orthogonal.</summary>
<details>
  <summary>My solution</summary>
</details>
<hr>