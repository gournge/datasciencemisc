# Quant interview

## 0xfdf x.com thread

Since my knowledge is limited, in my solutions I used Wikipedia, Rudin's Princples of Mathematical Analysis, Axler's Linear Algebra Done Right and James' An Introduction to Statistical Learning. 

Here is the [link to the X.com thread](https://x.com/0xfdf/status/1815166904010506620). 

The questions:

Question 1: I tell you that I am regressing $Y$ on $X$. Describe what I probably mean as rigorously and completely as you can, without sacrificing generality.

<details>
  <summary>My solution</summary>

*General linear model* or *generalized multivariate regression* (not: multiple linear regression) is a compact way of expressing many *multiple linear regressions* at the same time. So we can simply analyze the case of a multiple linear regression: one-dimensional response variable $Y$ and multidimensional predictor $X$.

I assume 
$X \in \mathbf{R}^{n\times p}$ and
$Y \in \mathbf{R}^{n\times 1}$
where $n$ is the number of observations and $p < n$ is the number of coefficients used for estimations.

- $Y$ is the dependent variable or response 
- $X$ is the independent variable or predictor or feature

$$
2+2 =4
$$

$$
  \begin{bmatrix}
  y_1\\
  y_2\\
  \vdots\\
  y_n
  \end{bmatrix}

  = 

  \begin{bmatrix}
  x_{1, 1} & x_{1, 2} & \ldots & x_{1, p} \\
  x_{2, 1} & \ddots & & \\  
  \vdots & & & \\
  x_{n, 1} & \ldots & & x_{n, p}\\
  \end{bmatrix}

  \begin{bmatrix}
  b_1\\
  b_2\\
  \vdots\\
  b_p
  \end{bmatrix}

  + 

  \begin{bmatrix}
  e_1\\
  e_2\\
  \vdots\\
  e_n
  \end{bmatrix}
$$

so in matrix form: $Y = X \beta + e$. The *Ordinary Least Squares* method wishes to find parameters $\beta = (\beta_{1}, \beta_{2} \ldots \beta_{p})^T$ that minimize the norm of the vector of residuals $e = (e_1, \ldots e_n)^T$.

So again, formally problem of regression is 

$$
\min_{\beta \in \mathbf{R}^p}{e^Te} = \min_{\beta \in \mathbf{R}^p}{e_1 ^ 2 + \ldots e_n ^ 2}
$$

where $e = Y - X\beta$. Let's examine the quantity $e^Te$:

$$
\begin{aligned}
e^Te &= (Y - X\beta)^T(Y - X\beta) \\ 
     &= (Y^T - (X\beta)^T)(Y- X\beta)) \\
     &= (Y^T - \beta ^ T X^T)(Y- X\beta)) \\
     &= Y^T Y - Y^T X \beta - \beta^T X^T Y  + \beta ^ T X^T X \beta 
\end{aligned}
$$

The optimum is a solution of 

$$
\begin{aligned}
0 = \frac{\partial (e^Te)}{\partial \beta} &= \frac{\partial (Y^T Y - Y^T X \beta - \beta^T X^T Y  + \beta ^ T X^T X \beta)} {\partial\beta} \\
                            &= -\frac{\partial (Y^T X \beta)}{\partial\beta} - \frac{\partial(\beta^T X^T Y )}{\partial\beta} + \frac{\partial(\beta ^ T X^T X \beta)}{\partial\beta} \\
                            &= -2X^TY+2X^TX\beta
\end{aligned}
$$

Hence $\hat{\beta} = (X^TX)^{-1} X^TY$. Now does $\hat{\beta}$ achieve the minimum or maximum? Since $e^Te = \sum_{i=1}^{n} e_i^2 \geq 0, \hat{\beta}$ *minimizes* the sum of squared residuals (squared $e_i$-s.). I think this gives us a hint why we can't do regression on complex variables (since $e^Te$ could be negative.) 

Now why $\frac{\partial Y^TX\beta}{\partial \beta} = \frac{\partial \beta^T X^T Y}{\partial \beta} = X^TY$? And what about the other term?

- First, we have to explicitly say that we are using the *denominator notation*. One of the differences compared to *numerator notation* can be seen in a case where $y \in \mathbf{R}, X \in \mathbf{R}^{p \times q}$ (so $X$ has $p$ rows and $q$ columns, and $\frac{\partial y}{\partial X} \in \mathbf{R}^{q \times p}$ with denominator notation).

  - So for example:

    - Denominator notation:

      $$
      \begin{aligned}
      \frac{\partial y}{\partial X} &= 
        \begin{bmatrix}
        \frac{\partial y}{\partial X_{1, 1}} & \frac{\partial y}{\partial X_{2, 1}} & \ldots & \frac{\partial y}{\partial X_{p, 1}} \\
        \frac{\partial y}{\partial X_{2, 1}} & \ddots & & \\  
        \vdots & & & \\
        \frac{\partial y}{\partial X_{1, q}} & \ldots & & \frac{\partial y}{\partial X_{q, p}}\\
        \end{bmatrix}
      \end{aligned}
      $$


    - Numerator notation:

      $$
      \begin{aligned}
      \frac{\partial y}{\partial X} &= 
        \begin{bmatrix}
        \frac{\partial y}{\partial X_{1, 1}} & \frac{\partial y}{\partial X_{1, 2}} & \ldots & \frac{\partial y}{\partial X_{1, p}} \\
        \frac{\partial y}{\partial X_{2, 1}} & \ddots & & \\  
        \vdots & & & \\
        \frac{\partial y}{\partial X_{q, 1}} & \ldots & & \frac{\partial y}{\partial X_{p, q}}\\
        \end{bmatrix}
      \end{aligned}
      $$

  - Put simply, $\text{numerator notation of M} = \text{(denominator notation of M)}^T$
  - Hence numerator notation can be seen as more intuitive (I have no idea why denominator notation was used here in Wikipedia)

- Second, we can use the definition of differentiation of a matrix. 
  - **Definition:** *[p.211 Rudin, 2024]*. Consider an open subset $E \subset \mathbf{R}^m, f : E \to \mathbf{R}^n$. $A \in \mathbf{R}^{m \times n}$ is said to be the derivative of $f$ at point $x \in \mathbf{R}^m$ iff $$ \lim_{h \to 0}{\frac{|f(x + h) - f(x) - Ah|}{|h|}} = 0\\ $$ We denote that fact with $$ f^{\prime}(x) = A $$ Of course $h \in \mathbf{R}^m$.
    - Rudin uses the numerator notation.
  - In the following remark, it is proved that $A = A^\prime (x) (= \frac{\partial (Ax)}{\partial x})$ using the fact that $A$ is a linear transformation: $$ \begin{aligned} \lim_{h \to 0}{\frac{|A(x + h) - A(x) - A(h)|}{|h|}} &= \lim_{h \to 0}{\frac{|A(x) + A(h) - A(x) - A(h)|}{|h|}} = 0 \end{aligned} $$ We have to note that the definition of $A$ as a matrix of real numbers and of $A$ as a linear transformation $A : \mathbf{R^m} \to \mathbf{R^n}$ is used here interchangeably and I have no idea if that is valid or problematic.
- Finally, if in the numerator notation $\frac{\partial (Ax)}{\partial x} = A$, then in the denominator notation $\frac{\partial (Ax)}{\partial x} = A^T$, we can conclude that in denominator notation (one used in Wikipedia's page on Multiple Linear Regression):
  - $\frac{\partial( Y^TX\beta)}{\partial \beta} = (Y^TX)^T = X^T (Y^T)^T = X^TY$, and similarly for the other term. 
    - **Remark:** Here we used the fact that if $A, B$ are matrices such that $AB$ makes sense, $(AB)^T = B^T A^T$. Let's prove it:
      - Instead of direct computation, we could use the fact that $A, B$ are just matrices of some actual linear transformations $X \in \mathcal{L}(V, W), Y \in \mathcal{L}(W, U)$, i.e. $\mathcal{M}(X) = A, \mathcal{M}(Y) = B$. (the L notation represents the space of all linear transformations between two vector spaces.)
      - From Axler's LADR (and following his notation): $$\mathcal{M}(S^{\prime}) = \mathcal{M}(S)^T$$ for all linear transformations $S$. $S^{\prime}$ denotes the dual transformation to $S$; $S^{\prime}(\psi) = \psi \circ S, \psi \in \mathcal{L}(V, F)$. We write $V^{\prime} = \mathcal{L}(V, F)$, which is the *Dual Space* of so called *linear functionals* with regards to vector space $S$.
      - In this spirit, the dual to the composition of two linear transformations $X, Y$, for all linear functionals $\psi \in \mathcal{L}(U^{\prime}, V^{\prime})$ is $(XY)^{\prime} (\psi) = \psi \circ (XY) = (\psi \circ X) \circ Y  = Y ^ {\prime} (\psi \circ X) = (Y ^ {\prime} X ^ {\prime}) (\psi)$.
      - We can represent this fact using matrix notation of these transformations: $$\begin{aligned} (AB)^T &= (\mathcal{M}(XY))^T = \mathcal{M}((XY)^{\prime}) = \mathcal{M}(Y^{\prime} X ^ {\prime}) = \mathcal{M}(Y ^ {\prime}) \mathcal{M}(X ^ {\prime}) \\ &= (\mathcal{M}(Y))^T (\mathcal{M}(X))^T = B^T A^T \end{aligned}$$
      - Note: in between the lines we use the fact that $S \in \mathcal{L}(V, W) \implies S^{\prime} \in \mathcal{L}(W^{\prime}, V^{\prime})$ 

Let's analyze the second term, $\frac{\partial(\beta ^ T X^T X \beta)}{\partial\beta}$:
- Rudin 2024 proved the chain rule for $f : \mathbf{R}^m \to \mathbf{R}^n, g : \mathbf{R}^n \to \mathbf{R}^k$, with appropriate conditions similar to the original derivative definition displayed above. 
- Let $f(t) = t ^ T t, \space g(t) = X t$ with $f : \mathbf{R}^{n \times 1} \to \mathbf{R}^{1 \times 1}, g : \mathbf{R}^{n \times 1} \to \mathbf{R}^{n \times 1}$ so $t \in \mathbf{R}^{n \times 1}$
- We have $$\begin{aligned} \frac{\partial(\beta ^ T X^T X \beta)}{\partial\beta} = \frac{\partial(f(g(\beta)))}{\partial\beta} &= \frac{\partial(f(g(\beta)))}{\partial (g(\beta))} \frac{\partial(g(\beta))}{\partial\beta} \\ &= () (X^T) \\  \end{aligned}$$

TODO: how to proceed with the derivative?

</details>
<hr>
Question 2: I have a univariate linear model of $X$ and $Y$, both $n \times 1$ column vectors. Describe what happens when I try to estimate this model with an exact copy of $X$ added as a second exogenous variable, and why, as rigorously and completely as you can.
<details>
  <summary>My solution</summary>
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