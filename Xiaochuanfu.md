![](http://latex.codecogs.com/gif.latex?\\frac{\\partial J}{\\partial \\theta_k^{(j)}}=\\sum_{i:r(i,j)=1}{\\big((\\theta^{(j)})

 

 

 

 

 

 

 

 

 

 

 

 <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

 

[2018/11/5-2018/11/11](#20181105-20181111)

- [本周进展](#%E6%9C%AC%E5%91%A8%E8%BF%9B%E5%B1%95)

- [遇到问题](#%E9%81%87%E5%88%B0%E9%97%AE%E9%A2%98)

- [下周计划](#%E4%B8%8B%E5%91%A8%E8%AE%A1%E5%88%92)

```

```

# 2018/11/05-2018/11/11

## 本周进展

- [x] 之前工作介绍

  之前主要是在阅读的论文"A general framework for Tucker factorization on Heterogeneous platforms"，论文中基于Jacobi型的Alternating Least Squares算法做了修改，使得算法的并行效率提高，但论文中并未对所提出算法--GTA算法进行过多的理论分析，例如修改后的算法解决的问题与原问题是否等价？算法的收敛性和收敛速度分析？我主要想理论证明GTA算法的收敛性，以及对其收敛速度进行分析，在这个过程中，我发现ALS算法的收敛性并没有足够的理论支撑，尤其是Jacobi型的ALS算法，所以证明GTA算法的收敛性是有一定难度的。通过MATLAB重现论文中的GTA算法，我发现算法在空间的角度收敛的。

  ​        问题描述：设
  $$
  X\in\mathbb{R}^{I\times J\times K}
  $$
  为一三阶张量，其Tucker分解的计算可由如下张量低秩逼近问题求解
  $$
  \min\limits_{G\in\mathbb{R}^{P\times Q\times R},A\in\mathbb{R}^{I\times P},B\in\mathbb{R}^{J\times Q},C\in\mathbb{R}^{K\times R}}\frac{1}{2}\|X - G\times_{1}A\times_{2}B\times_{3}C\|_{F}^{2}
  $$
  其中G为核张量，A, B, C 分别为列正交的因子矩阵，求解上述优化问题已经有一些算法，主要有HOSVD，HOOI，但上述算法的并行效率不高，而论文中提出的GTA算法为使得算法的并行效率高，所以将上述优化问题等价为
  $$
  \min\limits_{G\in\mathbb{R}^{P\times Q\times R},A\in\mathbb{R}^{I\times P},B\in\mathbb{R}^{J\times Q},C\in\mathbb{R}^{K\times R}}\frac{1}{2}\|X-G\times_{1}A\times_{2}B\times_{3}C\|_{F}^{2}
  $$
  其与原优化问题的不同是，将列正交的约束取消了，这样做的好处是在设计算法的时候可以并行处理。

  **GTA算法：**

  ------

  **输入：**张量
  $$
  X\in\mathbb{R}^{I\times J\times K}
  $$
  初始迭代列正交因子矩阵
  $$
  A_{0}\in\mathbb{R}^{I\times P},\ B_{0}\in\mathbb{R}^{J\times Q},\ C_{0}\in\mathbb{R}^{K\times R}
  $$
  **输出：**列正交因子矩阵
  $$
  A\in\mathbb{R}^{I\times P},\ B\in\mathbb{R}^{J\times Q},\ C\in\mathbb{R}^{K\times R}
  $$
   核张量
  $$
  G\in\mathbb{R}^{P\times Q\times R}；
  $$

  ------

  $$
  A = A_{0},\ B = B_{0},\ C = C_{0}
  $$

  **Repeat**

  $$
  G = X\times_{1}A^{T}\times_{2}B^{T}\times_{3}C^{T}
  $$

  $$
  A = X_{(1)}(G_{(1)}(C\bigotimes B)^{T})^{T}(G_{(1)}G_{(1)}^{T})^{-1}
  $$

  $$
  B = X_{(2)}(G_{(2)}(C\bigotimes A)^{T})^{T}(G_{(2)}G_{(2)}^{T})^{-1}
  $$

  $$
  C = X_{(3)}(G_{(3)}(B\bigotimes A)^{T})^{T}(G_{(3)}G_{(3)}^{T})^{-1}
  $$

  $$
  A = QR
  $$

  $$
  A = A
  $$

  $$
  B=QR
  $$

  ​                                                                  
  $$
  B=B
  $$

  $$
  C = QR
  $$

  $$
  C = C
  $$

  计算目标函数值： 
  $$
  Obejection = \frac{1}{2}\|X - G\times_{1}A\times_{2}B\times_{3}C\|_{F}^{2}
  $$
  **Until** 

  目标函数值达到停机准则；

  ------

  其中在更新因子矩阵
  $$
  A,\ B,\ C
  $$
   时可以并行处理，并且在每个因子矩阵的计算过程中可以将因子矩阵的行并行处理。从算法伪代码中易得到使用Jacobi型ALS算法解决此优化问题的迭代格式。

  另外，GTA算法是针对张量的Tucker分解的计算，我认为GTA算法思想也可以应用在矩阵SVD计算上，由于GTA算法的并行效率高，所以理论上可以使得矩阵SVD的计算并行，并且局部收敛性可以保证。对应的优化问题如下：
$$
\min\limits_{G\in\mathbb{R}^{R\times R},A\in\mathbb{R}^{I\times R},B\in\mathbb{R}^{J\times R}}\frac{1}{2}\|X - G\times_{1}A\times_{2}B\|_{F}^{2}=\frac{1}{2}\|X - AGB^{T}\|_{F}^{2}
$$
​       由于求解矩阵低秩逼近问题等价于求解矩阵SVD，所以当采取并行的方式计算出上述优化问题的解为A, G,B              再利用约化QR分解算法将矩阵A, B 进行正交化，即
$$
A = Q_{A}R_{A};\\
  B = Q_{B}R_{B};
$$
 令
$$
\hat{G} = R_{A}GR_{B}^{T}\in\mathbb{R}^{R\times R}
$$
 ，再对R阶矩阵
$$
\hat{G}
$$
进行SVD即可得到X的SVD，其算法伪代码如下：

------

**输入：**矩阵
$$
X\in\mathbb{R}^{I\times J}
$$
初始迭代列正交因子矩阵
$$
A_{0}\in\mathbb{R}^{I\times R},\ B_{0}\in\mathbb{R}^{J\times R}
$$
**输出：**矩阵$X$的SVD因子矩阵，
$$
A,\ B,\ \Sigma 
$$

------

$$
A = A_{0},\ B = B_{0}
$$

**Repeat**

$$
G = A^{T}XB
$$

$$
A = XBG^{-1}；
$$

$$
B = X^{T}AG^{-1}；
$$

$$
A = QR；
$$

$$
A = Q；
$$

$$
B = QR；
$$

$$
B = Q；
$$

计算目标函数值：
$$
Objection = \frac{1}{2}\|X - AGB^{T}\|^{2}_{F}
$$
；

**Until**

目标函数值达到停机准则；

计算G 的SVD，即
$$
G = U\Sigma V^{T}；
$$

$$
A = AU；
$$

$$
B = BV；
$$

$$
\Sigma；
$$

------

 对于计算低秩矩阵的SVD或者只要求计算前R个奇异值的问题，即
$$
R<<\min\{I,J\}
$$
 的情形，在更新因子矩阵
$$
A,\ B
$$
时，若将因子矩阵
$$
A,\ B
$$
的行并行处理，那么等价于计算I+J个R阶线性方程组，那么，每个迭代步所需的计算量包括求解I+J$个R阶线性方程组，两次QR分解以及更新G 的计算量，计算量大概为
$$
O(RIJ) + O(R^{2}J) + O((I+J)R^{2}) + O(RI^{2}) + O(RJ^{2})
$$
假设迭代步数为O(I+J),那么算法的总计算量为
$$
O(\max\{RIJ^{2},\ RI^{2}J\})
$$
 ，若使用Omax\{I,J\}个处理器计算I+J个R阶线性方程组，在理想加速比的情况下可以减少计算量级。

根据算法伪代码，可以得到因子矩阵A, B 的迭代格式，并且可以证明R(A),R(B) 是分别收敛到矩阵X 的前R个左奇异向量组成的子空间和前R个右奇异向量组成的子空间的，收敛速度为Q线性收敛。

- [x] 完成项目

  - GTA算法思想中对应的优化问题与原优化问题的等价性已证；
  - GTA算法的显式迭代格式已能给出，并能证明最优解是由因子矩阵的列空间决定的；
  - 使用MATLAB做数值实验能够说明GTA算法的空间收敛性；
  - GTA算法思想解决矩阵SVD计算的局部收敛性已能证明，并且收敛速度线性收敛；
  - 使用MATLAB验证过此方法解决矩阵SVD计算的正确性。
- [ ] 未完成项目
  - GTA算法的收敛性为能证明；
  - GTA算法思想解决矩阵SVD计算的并行代码未能写出，所以并行效率究竟如何并不知道。

## 遇到问题

* 问题一：能不能设计出并行算法解决最近新提出的张量分解如tensor train分解，t-SVD等？
* 问题二：张量的CUR分解中对slices或者fibers的选择问题是如何解决的？

## 下周计划

1. 任务一：完成最优化算法和理论课的上机作业；
2. 任务二：练习并行与分布式计算课上的代码，并加强编程能力；
3. 任务三：上述问题的数值结果整理出来。
