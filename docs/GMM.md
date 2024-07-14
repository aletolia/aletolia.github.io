!!! note 
     原文地址：https://blog.csdn.net/yuanmiyu6522/article/details/125211015?spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-10-125211015-blog-19182013.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-10-125211015-blog-19182013.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=9

## 一个例子

高斯混合模型（Gaussian Mixed Model）指的是多个高斯分布函数的线性组合，理论上 GMM 可以拟合出任意类型的分布，通常用于解决同一集合下的数据包含多个不同的分布的情况（或者是同一类分布但参数不一样，或者是不同类型的分布，比如正态分布和伯努利分布）。

如图 1，图中的点在我们看来明显分成两个聚类。这两个聚类中的点分别通过两个不同的正态分布随机生成而来。但是如果没有 GMM，那么只能用一个的二维高斯分布来描述图 1 中的数据。图 1 中的椭圆即为二倍标准差的正态分布椭圆。这显然不太合理，毕竟肉眼一看就觉得应该把它们分成两类。

![](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwMzAyMTc1NDQyMjcy?x-oss-process=image/format,png)

图 1

这时候就可以使用 GMM 了！如图 2，数据在平面上的空间分布和图 1 一样，这时使用两个二维高斯分布来描述图 2 中的数据，分别记为 $\mathcal{N}\left(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1\right)$ 和 $\mathcal{N}\left(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2\right)$. 图中的两个椭圆分别是这两个高斯分布的二倍标准差椭圆。可以看到使用两个二维高斯分布来描述图中的数据显然更合理。实际上图中的两个聚类的中的点是通过两个不同的正态分布随机生成而来。如果将两个二维高斯分布 $\mathcal{N}\left(\mu_1, \boldsymbol{\Sigma}_1\right)$ 和 $\mathcal{N}\left(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2\right)$ 合成一个二维的分布，那么就可以用合成后的分布来描述图 2 中的所有点。最直观的方法就是对这两个二维高斯分布做线性组合，用线性组合后的分布来描述整个集合中的数据。这就是高斯混合模型 (GMM)。

![](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwMzAyMTc1NTQ5ODc3?x-oss-process=image/format,png)

## 高斯混合模型 (GMM)

设有随机变量 $\boldsymbol{X}$ ，则混合高斯模型可以用下式表示:

$$
\mathrm{p}(\boldsymbol{x})=\sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)
$$

其中 $\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)$ 称为混合模型中的第 k 个分量 (component)。如前面图 2 中的例子，有两个聚类，可以用两个二维高斯分布来表示，那么分量数 $\mathrm{K}=2 . \pi_{\mathrm{k}}$ 是混合系数 (mixture coefficient)，且满足:

$$
\begin{aligned}
& \sum_{k=1}^K \pi_k=1 \\
& 0 \leq \pi_k \leq 1
\end{aligned}
$$

实际上，可以认为 $\pi_{\mathrm{k}}$ 就是每个分量 $\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)$ 的权重。

## GMM 的应用

GMM 常用于聚类。如果要从 GMM 的分布中随机地取一个点的话，实际上可以分为两步：首先随机地在这 K 个 Component 之中选一个，每个 Component 被选中的概率实际上就是它的系数 $\pi_{\mathrm{k}}$ ，选中 Component 之后，再单独地考虑从这个 Component 的分布中选取一个点就可以了一这里已经回到了普通的 Gaussian 分布，转化为已知的问题。

将 GMM 用于聚类时，假设数据服从混合高斯分布 (Mixture Gaussian Distribution)，那么只要根据数据推出 GMM 的概率分布来就可以了; 然后 GMM 的 K 个 Component 实际上对应 K 个 cluster。根据数据来推算概率密度通常被称作 density estimation。特别地，当我已知 (或假定) 概率密度函数的形式，而要估计其中的参数的过程被称作『参数估计』。

例如图 2 的例子，很明显有两个聚类，可以定义 $\mathrm{K}=2$. 那么对应的 GMM 形式如下:

$$
\mathrm{p}(\boldsymbol{x})=\pi_1 \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1\right)+\pi_2 \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2\right)
$$

上式中末知的参数有六个: $\left(\pi_1, \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1 ; \pi_2, \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2\right)$. 之前提到 GMM 聚类时分为两步，第一步是随机地在这 K 个分量中选一个，每个分 量被选中的概率即为混合系数 $\pi_{\mathrm{k}}$. 可以设定 $\pi_1=\pi_2=0.5$ ，表示每个分量被选中的概率是 0.5 ，即从中抽出一个点，这个点属于第一类 的概率和第二类的概率各占一半。但实际应用中事先指定 $\pi_{\mathrm{k}}$ 的值是很笨的做法，当问题一般化后，会出现一个问题: 当从图 2 中的集合随机选取一个点，怎么知道这个点是来自 $\mathrm{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1\right)$ 还是 $\mathrm{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2\right)$ 呢? 换言之怎么根据数据自动确定 $\pi_1$ 和 $\pi_2$ 的值? 这就是 GMM 参数估计的问题。要解决这个问题，可以使用 EM 算法。通过 EM 算法，我们可以迭代计算出 GMM 中的参数：$\left(\pi_{\mathrm{k}}, \boldsymbol{x}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)$

## GMM 参数估计过程

### GMM 的贝叶斯理解

在介绍 GMM 参数估计之前，先改写 GMM 的形式，改写之后的 GMM 模型可以方便地使用 EM 估计参数。GMM 的原始形式如下:

$$
\mathrm{p}(\boldsymbol{x})=\sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)\tag{1}
$$

前面提到 $\pi_{\mathrm{k}}$ 可以看成是第 k 类被选中的概率。我们引入一个新的 K 维随机变量 $\boldsymbol{z} . \mathrm{z}_{\mathrm{k}}(1 \leq \mathrm{k} \leq \mathrm{K})$ 只能取 0 或 1 两个值； $\mathrm{z}_{\mathrm{k}}=1$ 表示第 k 类被选中的概率，即: $\mathrm{p}\left(\mathrm{z}_{\mathrm{k}}=1\right)=\pi_{\mathrm{k}}$; 如果 $\mathrm{z}_{\mathrm{k}}=0$ 表示第 k 类没有被选中的概率。更数学化一点， $\mathrm{z}_{\mathrm{k}}$ 要满足以下两个条件:

$$
\begin{array}{r}
\mathrm{z}_{\mathrm{k}} \in\{0,1\} \\
\sum_{\mathrm{K}} \mathrm{z}_{\mathrm{k}}=1
\end{array}
$$

例如图 2 中的例子，有两类，则 $\boldsymbol{z}$ 的维数是 2 . 如果从第一类中取出一个点，则 $\boldsymbol{z}=(1,0)$; 如果从第二类中取出一个点，则 $\boldsymbol{z}=(0,1)$. $\mathrm{z}_{\mathrm{k}}=1$ 的概率就是 $\pi_{\mathrm{k}}$ ，假设 $\mathrm{z}_{\mathrm{k}}$ 之间是独立同分布的 (iid)，我们可以写出 $\boldsymbol{z}$ 的联合概率分布形式，就是连乘:

$$
\mathrm{p}(\boldsymbol{z})=\mathrm{p}\left(\mathrm{z}_1\right) \mathrm{p}\left(\mathrm{z}_2\right) \ldots \mathrm{p}\left(\mathrm{z}_{\mathrm{K}}\right)=\prod_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}}^{\mathrm{z}_{\mathrm{k}}}\tag{2}
$$

因为 $z_k$ 只能取 0 或 1 ，且 $\boldsymbol{z}$ 中只能有一个 $z_k$ 为 1 而其它 $z_j(\mathrm{j} \neq \mathrm{k})$ 全为 0 ，所以上式是成立的。

图 2 中的数据可以分为两类，显然，每一類中的数据都是服从正态分布的。这个叙述可以用条件概率来表示:

$$
\mathrm{p}\left(\boldsymbol{x} \mid \mathrm{z}_{\mathrm{k}}=1\right)=\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)
$$

即第 k 类中的数据服从正态分布。进而上式有可以写成如下形式:

$$
\mathrm{p}(\boldsymbol{x} \mid \boldsymbol{z})=\prod_{\mathrm{k}=1}^{\mathrm{K}} \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)^{\mathrm{z}_{\mathrm{k}}}\tag{3}
$$

上面分别给出了 $\mathrm{p}(\boldsymbol{z})$ 和 $\mathrm{p}(\boldsymbol{x} \mid \boldsymbol{z})$ 的形式，根据条件概率公式，可以求出 $\mathrm{p}(\boldsymbol{x})$ 的形式:

$$
\begin{aligned}
\mathrm{p}(\boldsymbol{x}) & =\sum_{\boldsymbol{z}} \mathrm{p}(\boldsymbol{z}) \mathrm{p}(\boldsymbol{x} \mid \boldsymbol{z}) \\
& =\sum_{\boldsymbol{z}}\left(\prod_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}}^{\mathrm{zk}} \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)^{\mathrm{z}_{\mathrm{k}}}\right) \\
& =\sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)
\end{aligned}\tag{4}
$$

(注: 上式第二个等号，对 $z$ 求和，实际上就是 $\sum_{\mathrm{k}=1}^{\mathrm{R}}$ 。又因为对某个 k ，只要 $\mathrm{i} \neq \mathrm{k}$ ，则有 $\mathrm{z}_{\mathrm{i}}=0$ ，所以 $\mathrm{z}_{\mathrm{k}}=0$ 的项为 1 ，可省略，最终得到第三个等号)

可以看到 GMM 模型的 (1) 式与 (4) 式有一样的形式，且 (4) 式中引入了一个新的变量 $\boldsymbol{z}$ ，通常称为隐含变量 (latent variable)。对于图 2 中的数据，『隐含』的意义是: 我们知道数据可以分成两类，但是随机抽取一个数据点，我们不知道这个数据点属于第一类还是第二类，它的归属我们观察不到，因此引入一个隐含变量 $z$ 来描述这个现象。

注意到在贝叶斯的思想下， $\mathrm{p}(\boldsymbol{z})$ 是先验概率， $\mathrm{p}(\boldsymbol{x} \mid \boldsymbol{z})$ 是似然概率，很自然我们会想到求出后验概率 $\mathrm{p}(\boldsymbol{z} \mid \boldsymbol{x})$ :

$$
\begin{aligned}
\gamma\left(\mathrm{z}_{\mathrm{k}}\right) & =\mathrm{p}\left(\mathrm{z}_{\mathrm{k}}=1 \mid \boldsymbol{x}\right) \\
& =\frac{\mathrm{p}\left(\mathrm{z}_{\mathrm{k}}=1\right) \mathrm{p}\left(\boldsymbol{x} \mid \mathrm{z}_{\mathrm{k}}=1\right)}{\mathrm{p}\left(\boldsymbol{x}, \mathrm{z}_{\mathrm{k}}=1\right)} \\
& =\frac{\mathrm{p}\left(\mathrm{z}_{\mathrm{k}}=1\right) \mathrm{p}\left(\boldsymbol{x} \mid \mathrm{z}_{\mathrm{k}}=1\right)}{\sum_{\mathrm{j}=1}^{\mathrm{K}} \mathrm{p}\left(\mathrm{z}_{\mathrm{j}}=1\right) \mathrm{p}\left(\boldsymbol{x} \mid \mathrm{z}_{\mathrm{j}}=1\right)} \text { (全概率公式) } \\
& =\frac{\pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_{\boldsymbol{k}}\right)}{\sum_{\mathrm{j}=1}^{\mathrm{K}} \pi_{\mathrm{j}} \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_{\mathrm{j}}\right)} \text { (结合 (3)、(4)) }
\end{aligned}\tag{5}
$$

（第 2 行，贝叶斯定理。关于这一行的分母，很多人有疑问，应该是 $\mathrm{p}\left(\boldsymbol{x}, \mathrm{z}_{\mathrm{k}}=1\right)$ 还是 $\mathrm{p}(\boldsymbol{x})$ ，按昭正常写法，应该是 $\mathrm{p}(\boldsymbol{x})$ 。但是为了强 调 $\mathrm{z}_{\mathrm{k}}$ 的取值，有的书会写成 $\mathrm{p}\left(\boldsymbol{x}, \mathrm{z}_{\mathrm{k}}=1\right)$ ，比如李航的《统计学习方法》，这里就约定 $\mathrm{p}(\boldsymbol{x})$ 与 $\mathrm{p}\left(\boldsymbol{x}, \mathrm{z}_{\mathrm{k}}=1\right)$ 是等同的)
上式中我们定义符号 $\gamma\left(\mathrm{z}_{\mathrm{k}}\right)$ 来表示来表示第 k 个分量的后验概率。在贝叶斯的观点下， $\pi_{\mathrm{k}}$ 可视为 $\mathrm{z}_{\mathrm{k}}=1$ 的先验概率。
上述内容改写了 GMM 的形式，并引入了隐含变量 $\boldsymbol{z}$ 和已知 $\boldsymbol{x}$ 后的的后验概率 $\gamma\left(\mathrm{z}_{\mathrm{k}}\right)$ ，这样做是为了方便使用 EM 算法来估计 GMM 的参数。

### EM 算法估计 GMM 参数

EM 算法 (Expectation-Maximization algorithm) 分两步，第一步先求出要估计参数的粗略值，第二步使用第一步的值最大化似然函数。因此要先求出 GMM 的似然函数。

假设 $\boldsymbol{x}=\left\{\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_{\mathrm{N}}\right\}$ ，对于图 2， $\boldsymbol{x}$ 是图中所有点（每个点有在二维平面上有两个坐标，是二维向量，因此 $\boldsymbol{x}_1, \boldsymbol{x}_2$ 等都用粗体表示)。GMM 的概率模型如 (1) 式所示。GMM 模型中有三个参数需要估计，分别是 $\boldsymbol{\pi}, \boldsymbol{\mu}$ 和 $\boldsymbol{\Sigma}$. 将 (1) 式稍微改写一下:

$$
\mathrm{p}(\boldsymbol{x} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})=\sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)\tag{6}
$$

为了估计这三个参数，需要分别求解出这三个参数的最大似然函数。先求解 $\mu_{\mathrm{k}}$ 的最大似然函数。样本符合 iid，(6) 式所有样本连乘得到最大似然函数，对 (6) 式取对数得到对数似然函数，然后再对 $\mu_{\mathrm{k}}$ 求导并令导数为 0 即得到最大似然函数。

$$
0=-\sum_{\mathrm{n}=1}^{\mathrm{N}} \frac{\pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x}_{\mathrm{n}} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)}{\sum_{\mathrm{j}} \pi_{\mathrm{j}} \mathcal{N}\left(\boldsymbol{x}_{\mathrm{n}} \mid \boldsymbol{\mu}_{\mathrm{j}}, \boldsymbol{\Sigma}_{\mathrm{j}}\right)} \boldsymbol{\Sigma}_{\mathrm{k}}^{-1}\left(\boldsymbol{x}_{\mathrm{n}}-\boldsymbol{\mu}_{\mathrm{k}}\right)\tag{7}
$$

注意到上式中分数的一项的形式正好是 (5) 式后验概率的形式。两边同乘 $\boldsymbol{\Sigma}_{\mathrm{k}}$ ，重新整理可以得到:

$$
\boldsymbol{\mu}_{\mathrm{k}}=\frac{1}{\mathrm{~N}_{\mathrm{k}}} \sum_{\mathrm{n}=1}^{\mathrm{N}} \gamma\left(\mathrm{z}_{\mathrm{nk}}\right) \boldsymbol{x}_{\mathrm{n}}\tag{8}
$$

其中:

$$
\mathrm{N}_{\mathrm{k}}=\sum_{\mathrm{n}=1}^{\mathrm{N}} \gamma\left(\mathrm{z}_{\mathrm{nk}}\right)\tag{9}
$$

(8) 式和 (9) 式中， N 表示点的数量。 $\gamma\left(\mathrm{z}_{\mathrm{nk}}\right)$ 表示点 $\mathrm{n}\left(\boldsymbol{x}_{\mathrm{n}}\right)$ 属于聚类 k 的后验概率。则 $\mathrm{N}_{\mathrm{k}}$ 可以表示属于第 k 个聚类的点的数量。那么 $\boldsymbol{\mu}_{\mathrm{k}}$ 表示所有点的加权平均，每个点的权值是 $\sum_{\mathrm{n}=1}^{\mathrm{N}} \gamma\left(\mathrm{z}_{\mathrm{nk}}\right)$ ，跟第 k 个聚类有关。

同理求 $\boldsymbol{\Sigma}_{\mathrm{k}}$ 的最大似然函数，可以得到:

$$
\boldsymbol{\Sigma}_{\mathrm{k}}=\frac{1}{\mathrm{~N}_{\mathrm{k}}} \sum_{\mathrm{n}=1}^{\mathrm{N}} \gamma\left(\mathrm{z}_{\mathrm{nk}}\right)\left(\mathrm{x}_{\mathrm{n}}-\boldsymbol{\mu}_{\mathrm{k}}\right)\left(\mathrm{x}_{\mathrm{n}}-\boldsymbol{\mu}_{\mathrm{k}}\right)^{\mathrm{T}}\tag{10}
$$

最后剩下 $\pi_{\mathrm{k}}$ 的最大似然函数。注意到 $\pi_{\mathrm{k}}$ 有限制条件 $\sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}}=1$ ，因此我们需要加入拉格朗日算子:

$$
\ln \mathrm{p}(\boldsymbol{x} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})+\lambda\left(\sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}}-1\right)
$$

求上式关于 $\pi_{\mathrm{k}}$ 的最大似然函数，得到:

$$
0=\sum_{\mathrm{n}=1}^{\mathrm{N}} \frac{\mathcal{N}\left(\boldsymbol{x}_{\mathrm{n}} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)}{\sum_{\mathrm{j}} \pi_{\mathrm{j}} \mathcal{N}\left(\boldsymbol{x}_{\mathrm{n}} \mid \boldsymbol{\mu}_{\mathrm{j}}, \boldsymbol{\Sigma}_{\mathrm{j}}\right)}+\lambda \tag{11}
$$

上式两边同乘 $\pi_{\mathrm{k}}$ ，我们可以做如下推导:

$$
0=\sum_{\mathrm{n}=1}^{\mathrm{N}} \frac{\pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x}_{\mathrm{n}} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)}{\sum_{\mathrm{j}} \pi_{\mathrm{j}} \mathcal{N}\left(\boldsymbol{x}_{\mathrm{n}} \mid \boldsymbol{\mu}_{\mathrm{j}}, \boldsymbol{\Sigma}_{\mathrm{j}}\right)}+\lambda \pi_{\mathrm{k}}\tag{11.1}
$$

结合公式 (5)，(9)，可以将上式改写成:

$$
0=\mathrm{N}_{\mathrm{k}}+\lambda \pi_{\mathrm{k}}\tag{11.2}
$$

注意到 $\sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}}=1$ ，上式两边同时对 k 求和。此外 $\mathrm{N}_{\mathrm{k}}$ 表示属于第个聚类的点的数量 (公式 (9))。对 $\mathrm{N}_{\mathrm{k}}$, 从 $\mathrm{k}=1$ 到 $\mathrm{k}=\mathrm{K}$ 求和后，就是所有点的数量 N :

$$
\begin{gathered}
0=\sum_{k=1}^K \mathrm{~N}_{\mathrm{k}}+\lambda \sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_k \\
0=\mathrm{N}+\lambda
\end{gathered}\tag{11.3,11.4}
$$

从而可得到 $\lambda=-\mathrm{N}$ ，带入 (11.2)，进而可以得到 $\pi_{\mathrm{k}}$ 更简洁的表达式:

$$
\pi_{\mathrm{k}}=\frac{\mathrm{N}_{\mathrm{k}}}{\mathrm{N}}\tag{12}
$$

EM 算法估计 GMM 参数即最大化 (8)，(10) 和 (12)。需要用到 (5)，(8)，(10) 和 (12) 四个公式。我们先指定 $\boldsymbol{\pi}, \boldsymbol{\mu}$ 和 $\boldsymbol{\Sigma}$ 的初始值，带入 (5) 中计算出 $\gamma\left(\mathrm{z}_{\mathrm{nk}}\right)$ ，然后再将 $\gamma\left(\mathrm{z}_{\mathrm{nk}}\right)$ 带入 (8)，(10) 和 (12)，求得 $\pi_{\mathrm{k}}, \boldsymbol{\mu}_{\mathrm{k}}$ 和 $\boldsymbol{\Sigma}_{\mathrm{k}}$; 接着用求得的 $\pi_{\mathrm{k}}, \boldsymbol{\mu}_{\mathrm{k}}$ 和 $\boldsymbol{\Sigma} \boldsymbol{\Sigma} \mathrm{k}$ 再带入 (5) 得到新的 $\gamma\left(\mathrm{z}_{\mathrm{nk}}\right)$ ，再将更新后的 $\gamma\left(\mathrm{z}_{\mathrm{nk}}\right)$ 带入 (8)，(10) 和 (12)，如此往复，直到算法收敛。

## EM 算法

1. 定义分量数目 K ，对每个分量 k 设置 $\pi_{\mathrm{k}} ， \mu_{\mathrm{k}}$ 和 $\Sigma_{\mathrm{k}}$ 的初始值，然后计算 (6) 式的对数似然函数。
2. E step
   根据当前的 $\pi_{\mathrm{k}} 、 \boldsymbol{\mu}_{\mathrm{k}} 、 \boldsymbol{\Sigma}_{\mathrm{k}}$ 计算后验概率 $\gamma\left(\mathrm{z}_{\mathrm{nk}}\right)$
   $$
   \gamma\left(\mathrm{z}_{\mathrm{nk}}\right)=\frac{\pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x}_{\mathrm{n}} \mid \boldsymbol{\mu}_{\mathrm{n}}, \boldsymbol{\Sigma}_{\mathrm{n}}\right)}{\sum_{\mathrm{j}=1}^{\mathrm{K}} \pi_{\mathrm{j}} \mathcal{N}\left(\boldsymbol{x}_{\mathrm{n}} \mid \boldsymbol{\mu}_{\mathrm{j}}, \boldsymbol{\Sigma}_{\mathrm{j}}\right)}
   $$
3. M step
   根据 E step 中计算的 $\gamma\left(\mathrm{z}_{\mathrm{nk}}\right)$ 再计算新的 $\pi_{\mathrm{k}} 、 \boldsymbol{\mu}_{\mathrm{k}} 、 \boldsymbol{\Sigma}_{\mathrm{k}}$
   $$
   \begin{aligned}
   \boldsymbol{\mu}_{\mathrm{k}}^{\text {new }} & =\frac{1}{\mathrm{~N}_{\mathrm{k}}} \sum_{\mathrm{n}=1}^{\mathrm{N}} \gamma\left(\mathrm{z}_{\mathrm{nk}}\right) \boldsymbol{x}_{\mathrm{n}} \\
   \boldsymbol{\Sigma}_{\mathrm{k}}^{\text {new }} & =\frac{1}{\mathrm{~N}_{\mathrm{k}}} \sum_{\mathrm{n}=1}^{\mathrm{N}} \gamma\left(\mathrm{z}_{\mathrm{nk}}\right)\left(\boldsymbol{x}_{\mathrm{n}}-\boldsymbol{\mu}_{\mathrm{k}}^{\text {new }}\right)\left(\boldsymbol{x}_{\mathrm{n}}-\boldsymbol{\mu}_{\mathrm{k}}^{\text {new }}\right)^{\mathrm{T}} \\
   \pi_{\mathrm{k}}^{\text {new }} & =\frac{\mathrm{N}_{\mathrm{k}}}{\mathrm{N}}
   \end{aligned}
   $$

其中:

$$
\mathrm{N}_{\mathrm{k}}=\sum_{\mathrm{n}=1}^{\mathrm{N}} \gamma\left(\mathrm{z}_{\mathrm{nk}}\right)
$$

4. 计算 (6) 式的对数似然函数
   $$
   \ln \mathrm{p}(\boldsymbol{x} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})=\sum_{\mathrm{n}=1}^{\mathrm{N}} \ln \left\{\sum_{\mathrm{k}=1}^{\mathrm{K}} \pi_{\mathrm{k}} \mathcal{N}\left(\boldsymbol{x}_{\mathrm{k}} \mid \boldsymbol{\mu}_{\mathrm{k}}, \boldsymbol{\Sigma}_{\mathrm{k}}\right)\right\}
   $$
5. 检查参数是否收敛或对数似然函数是否收敛，若不收敛，则返回第 2 步。

GMM 聚类的可分性评价
使用 GMM 得到聚类结果后如何定量评价两个类别的可分性呢? 可以通过计算两个或多个类别分布的重疍度来评价模型的可分性。这里介
