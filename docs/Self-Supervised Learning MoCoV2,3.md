---
url: https://zhuanlan.zhihu.com/p/383661584
title: MoCo V2
date: 2024-03-07 20:08:42
tag: 
summary:
---
**Self-Supervised Learning**，又称为自监督学习，我们知道一般机器学习分为有监督学习，无监督学习和强化学习。 而 Self-Supervised Learning 是无监督学习里面的一种，主要是希望能够学习到一种**通用的特征表达**用于**下游任务 (Downstream Tasks)**。 其主要的方式就是通过自己监督自己。作为代表作的 kaiming 的 MoCo 引发一波热议， Yann Lecun 也在 AAAI 上讲 Self-Supervised Learning 是未来的大势所趋。所以在这个系列中，我会系统地解读 Self-Supervised Learning 的经典工作。

* **1 MoCo v2**

**论文名称：Improved Baselines with Momentum Contrastive Learning**

**论文地址：**

[https://arxiv.org/pdf/2003.04297.pdf](https://arxiv.org/pdf/2003.04297.pdf)

* **1.1 MoCo v2 的 Motivation**

上篇文章我们介绍了 MoCo 这个系列的第一版 MoCo v1，链接如下所示。

**MoCo v1 的方法其实可以总结为 2 点**

**(a)** 在 **[1 原始的端到端自监督学习方法]** 里面，Encoder $f_\textrm{q}$ 和 Encoder $f_\textrm{k}$ 的参数每个 step 都更新，这个问题在前面也有提到，因为 Encoder $f_\textrm{k}$ 输入的是一个 Batch 的 negative samples (N-1 个)，所以输入的数量不能太大，即 dictionary 不能太大，即 Batch size 不能太大。

现在的 Momentum Encoder $f_\textrm{Mk}$ 的更新是通过动量的方法更新的，不涉及反向传播，所以 $f_\textrm{Mk}$ 输入的负样本 (negative samples) 的数量可以很多，具体就是 Queue 的大小可以比较大，可以比 mini-batch 大，属于超参数。队列是逐步更新的在每次迭代时，当前 mini-batch 的样本入列，而队列中最老的 mini-batch 样本出列，那当然是负样本的数量越多越好了。这就是 Dictionary as a queue 的含义，即通过动量更新的形式，使得可以包含更多的负样本。而且 Momentum Encoder $f_\textrm{Mk}$ 的更新极其缓慢 (因为 $m=0.999$ 很接近于 1)，所以 Momentum Encoder $f_\textrm{Mk}$ 的更新相当于是看了很多的 Batch，也就是很多负样本。

**(b)** 在 **[2 采用一个较大的 memory bank 存储较大的字典]** 方法里面，所有样本的 representation 都存在 memory bank 里面，根据上文的描述会带来最新的 query 采样得到的 key 可能是好多个 step 之前的编码器编码得到的 key，因此丧失了一致性的问题。但是 MoCo 的每个 step 都会更新 Momentum Encoder，虽然更新缓慢，但是每个 step 都会通过 4 式更新一下 Momentum Encoder，这样 Encoder $f_\textrm{q}$ 和 Momentum Encoder $f_\textrm{Mk}$ 每个 step 都有更新，就解决了一致性的问题。

**SimCLR 的两个提点的方法**

今天介绍 MoCo 系列的后续工作：MoCo v2 和 v3。MoCo v2 是在 SimCLR 发表以后相继出来的，它是一篇很短的文章， 只有 2 页。在 MoCo v2 中，作者们整合 **SimCLR 中的两个主要提升方法**到 MoCo 中，并且验证了 SimCLR 算法的有效性。**SimCLR 的两个提点的方法就是：**

* 使用强大的数据增强策略，具体就是额外使用了 Gaussian Deblur 的策略和使用巨大的 Batch size，让自监督学习模型在训练时的每一步见到足够多的负样本 (negative samples)，这样有助于自监督学习模型学到更好的 visual representations。
* 使用预测头 Projection head。在 SimCLR 中，Encoder 得到的 2 个 visual representation 再通过 Prediction head ($g(.)$) 进一步提特征，预测头是一个 2 层的 MLP，将 visual representation 这个 2048 维的向量 $h_i,h_j$ 进一步映射到 128 维隐空间中，得到新的 representation $z_i,z_j$。利用 $z_i,z_j$ 去求 loss 完成训练，训练完毕后扔掉预测头，保留 Encoder 用于获取 visual representation。

关于 SimCLR 的详细解读欢迎参考下面的链接：

[科技猛兽：Self-Supervised Learning 超详细解读 (二)：SimCLR 系列](https://zhuanlan.zhihu.com/p/378953015)

SimCLR 的方法其实是晚于 MoCo v1 的。时间线如下：

MoCo v1 于 2019.11 发布于 arXiv，中了 CVPR 2020；
SimCLR v1 于 2020.02 发布于 arXiv，中了 ICML 2020；
MoCo v2 于 2020.03 发布于 arXiv，是一个技术报告，只有 2 页。
SimCLR v2 于 2020.06 发布于 arXiv，中了 NIPS 2020；

在 SimCLR v1 发布以后，MoCo 的作者团队就迅速地将 SimCLR 的两个提点的方法移植到了 MoCo 上面，想看下性能的变化，也就是 MoCo v2。结果显示，MoCo v2 的结果取得了进一步的提升并超过了 SimCLR v1，证明 MoCo 系列方法的地位。因为 MoCo v2 文章只是移植了 SimCLR v1 的技巧而没有大的创新，所以作者就写成了一个只有 2 页的技术报告。

* **1.2 MoCo 相对于 End-to-end 方法的改进**

MoCo v2 的亮点是不需要强大的 Google TPU 加持，仅仅使用 8-GPU 就能超越 SimCLR v1 的性能。End-to-end 的方法和 MoCo v1 的方法在本专栏的上一篇文章 **Self-Supervised Learning 超详细解读 (四)：MoCo 系列 (1)** 里面已经有详细的介绍，这里再简单概述下二者的不同，如下图 1,2,3 所示。

![](https://pic3.zhimg.com/v2-04664082d9f4107922d303ba53c1d5be_r.jpg)

![](https://pic1.zhimg.com/v2-8db083a9138be546aa9ac38974eb5e5c_r.jpg)

![](https://pic4.zhimg.com/v2-7b8430d1871900203b3547dc92d55f57_r.jpg)

**End-to-end 的方法 (原始的端到端自监督学习方法)：**一个 Batch 的数据假设有 $N$ 个 image，这里面有一个样本 query $\color{purple}{q}$ 和它所对应的正样本 $\color{crimson}{k^{+}}$ ， $\color{purple}{q}$ 和 $\color{crimson}{k^{+}}$ 来自同一张图片的不同的 Data Augmentation，这个 Batch 剩下的数据就是负样本 (negative samples)，如下图 3 所示。接着我们将这个 Batch 的数据同时输入给 2 个**架构相同但参数不同**的 Encoder $f_\textrm{q}$ 和 Encoder $f_\textrm{k}$ 。然后对两个 Encoder 的输出使用下式 1 所示的 Contrastive loss 损失函数使得 query $\color{purple}{q}$ 和正样本 $\color{crimson}{k^{+}}$ 的相似程度尽量地高，使得 query $\color{purple}{q}$ 和负样本 $\color{green}{k^{-}}$ 的相似程度尽量地低，通过这样来训练 Encoder $f_\textrm{q}$ 和 Encoder $f_\textrm{k}$，这个过程就称之为自监督预训练。训练完毕后得到的 Encoder 的输出就是图片的 visual representation。这种方法的缺点是：因为 Encoder $f_\textrm{q}$ 和 Encoder $f_\textrm{k}$ 的参数都是通过反向传播来更新的，所以 Batch size 的大小不能太大，否则 GPU 显存就不够了。所以，Batch size 的大小限制了负样本的数量，也限制了自监督模型的性能。

![](https://pic3.zhimg.com/v2-6b29a60dc533275a8d66d58ae7807a52_b.jpg)

$$
\begin{equation} \small \mathcal{L}_{q, k^+, \{k^-\}} = -\log \frac{\exp(q{\cdot}k^+  \tau)}{\exp(q{\cdot}k^+  \tau) + {\displaystyle\sum_{k^-}}\exp(q{\cdot}k^-  \tau)}. \end{equation} \tag{1}
$$

**MoCo 的方法：** 一个 Batch 的数据假设有 $N$ 个 image，这里面有一个样本 query $\color{purple}{q}$ 和它所对应的正样本 $\color{crimson}{k^{+}}$ ， $\color{purple}{q}$ 和 $\color{crimson}{k^{+}}$ 来自同一张图片的不同的 Data Augmentation，这个 Batch 剩下的数据就是负样本 (negative samples)。 接着我们只把 query $\color{purple}{q}$ 和正样本 $\color{crimson}{k^{+}}$ 输入给 2 个**架构相同但参数不同**的 Encoder $f_\textrm{q}$ 和 Momentum Encoder $f_\textrm{Mk}$ 。所有的负样本 $\color{green}{k^{-}}$ 都会保存在一个队列 Queue 里面。然后对两个 Encoder 的输出使用上式 1 所示的 Contrastive loss 损失函数使得 query $\color{purple}{q}$ 和正样本 $\color{crimson}{k^{+}}$ 的相似程度尽量地高，使得 query $\color{purple}{q}$ 和负样本 $\color{green}{k^{-}}$ 的相似程度尽量地低。在任意一个 Epoch 的任意一个 step 里面，我们只使用反向传播来更新 Encoder $f_\textrm{q}$ 的参数，然后通过 2 式的动量方法更新 Momentum Encoder $f_\textrm{Mk}$ 的参数。同时，队列删除掉尾部的一个 Batch 大小的负样本，再在头部进来一个 Batch 大小的负样本，完成这个 step 的队列的更新。这样，**队列的大小可以远远大于 Batch size 的大小了**，使得**负样本的数量可以很多**，提升了自监督训练的效果。而且，**队列和** $f_\textrm{Mk}$ **在每个 step 都会有更新**，没有 memory bank，也就**不会存在更新不及时导致的 $f_\textrm{q}$ 的更新和 memory bank 更新不一致**的问题。

$$
\theta_\textrm{k}\leftarrow m\theta_\textrm{k}+(1-m)\theta_\textrm{q}\tag{2}
$$

**FAQ：MoCo 方法里面这个队列 Queue 的内容是什么，是如何生成的？**

**答：**是负样本 $\color{green}{k^{-}}$ 通过 Momentum Encoder $f_\textrm{Mk}$ ( $f_\textrm{Mk}$ 采用 2 式的动量更新方法，而不是反向传播) 之后输出的值，它代表所有负样本的 visual representation。队列 Queue 的是 Batch size 的数倍大，且每个 step 都会进行一次 Dequeue 和 Enqueue 的操作更新队列。

* **1.3 MoCo v2 实验**

以上就是对 MoCo v1 的概述，v2 将 SimCLR 的两个提点的方法 (**a 使用预测头** **b 使用强大的数据增强策略**) 移植到了 MoCo v1 上面，实验如下。

**训练集：**ImageNet 1.28 张训练数据。

**评价手段：**

**(1) Linear Evaluation** (在 **Self-Supervised Learning 超详细解读 (二)：SimCLR 系列** 文章中有介绍，Encoder (ResNet-50) 的参数固定不动，在 Encoder 后面加分类器，具体就是一个 FC 层 + softmax 激活函数，使用全部的 ImageNet label 只训练分类器的参数，而不训练 Encoder 的参数)。看最后 Encoder + 分类器的性能。

**(2)** **VOC 目标检测** 使用 Faster R-CNN 检测器 (C4 backbone)，在 VOC 07+12 trainval set 数据集进行 End-to-end 的 Fine-tune。在 VOC 07 test 数据集进行 Evaluation。

**a 使用预测头结果**

预测头 Projection head 分类任务的性能只存在于无监督的预训练过程，在 **Linear Evaluation 和下游任务中都是被去掉的。**

Linear Evaluation 结果如下图 5 所示：

![](https://pic1.zhimg.com/v2-d3f901fe52011c9bf786a92c4ced6f04_r.jpg)

图中的 $\tau$ 就是式 1 中的 $\tau$ 。在使用预测头且 $\tau=0.07$ 时取得了最优的性能。

VOC 目标检测如下图 6 所示。在使用预测头且预训练的 Epoch 数为 800 时取得了最优的性能，AP 各项指标也超越了有监督学习 supervised 的情况。

![](https://pic3.zhimg.com/v2-bb27dbe29f50690c6e4e4dd2b2c3f18e_r.jpg)

**b 使用强大的数据增强策略结果**

如图 4 所示，对数据增强策略，作者在 MoCo v1 的基础上又添加了 blur augmentation，发现更强的色彩干扰作用有限。只添加 blur augmentation 就可以使得 ImageNet 分类任务的性能从 60.6 增长到 63.4，再加上预测头 Projection head 就可以使性能进一步涨到 67.3。从图 4 也可以看到：**VOC 目标检测的性能和 ImageNet 分类任务的性能没有直接的联系**。

**与 SimCLR v1 的对比**

如下图 7 所示为 MoCo v2 与 SimCLR v1 性能的直接对比结果。预训练的 Epochs 都取 200。如果 Batch size 都取 256，MoCo v2 在 ImageNet 有 67.5 的性能，超过了 SimCLR 的 61.9 的性能。即便 SimCLR 在更有利的条件下 (Batch size = 4096，Epochs=1000)，其性能 69.3 也没有超过 MoCo v2 的 71.1 的性能，证明了 MoCo 系列方法的地位。

![](https://pic4.zhimg.com/v2-fdf55b1771dae31041e9bc1f38f3749f_r.jpg)

**小结**

MoCo v2 把 **SimCLR 中的两个主要提升方法 (1 使用强大的数据增强策略，具体就是额外使用了 Gaussian Deblur 的策略 2 使用预测头 Projection head)** 到 MoCo 中，并且验证了 SimCLR 算法的有效性。最后的 MoCo v2 的结果更优于 SimCLR v1，证明 MoCo 系列自监督预训练方法的高效性。

* **2 MoCo v3**

**论文名称：An Empirical Study of Training Self-Supervised Vision Transformers**

**论文地址：**

[https://arxiv.org/pdf/2104.02057.pdf](https://arxiv.org/pdf/2104.02057.pdf)

* **2.1 MoCo v3 原理分析**

自监督学习模型一般可以分成 Generative 类型的或者 Contrastive 类型的。在 NLP 里面的自监督学习模型 (比如本专栏第 1 篇文章介绍的 BERT 系列等等) 一般是属于 Generative 类型的，通常把模型设计成 Masked Auto-encoder，即盖住输入的一部分 (Mask)，让模型预测输出是什么 (像做填空题)，通过这样的自监督方式预训练模型，让模型具有一个不错的预训练参数，且模型架构一般是个 Transformer。在 CV 里面的自监督学习模型 (比如本专栏第 2 篇文章介绍的 SimCLR 系列等等) 一般是属于 Contrastive 类型的，模型架构一般是个 Siamese Network (孪生神经网络)，通过数据增强的方式创造正样本，同时一个 Batch 里面的其他数据为负样本，通过使模型最大化样本与正样本之间的相似度，最小化与样本与负样本之间的相似度来对模型参数进行预训练，且孪生网络架构一般是个 CNN。

这篇论文的重点是将目前无监督学习最常用的对比学习应用在 ViT 上。作者给出的结论是：**影响自监督 ViT 模型训练的关键是：instability，即训练的不稳定性。**而这种训练的不稳定性所造成的结果**并不是训练过程无法收敛 (convergence)，而是性能的轻微下降 (下降 1%-3% 的精度)。**

首先看 MoCo v3 的具体做法吧。它的损失函数和 v1 和 v2 版本是一模一样的，都是 1 式：

$$
\mathcal{L}_{q, k^{+},\left\{k^{-}\right\}}=-\log \frac{\exp \left(q \cdot k^{+} / \tau\right)}{\exp \left(q \cdot k^{+} / \tau\right)+\sum_{k^{-}} \exp \left(q \cdot k^{-} / \tau\right)}
$$

那么不一样的是整个 Framework 有所差异，MoCo v3 的整体框架如下图 8 所示，这个图比论文里的图更详细地刻画了 MoCo v3 的训练方法，读者可以把图 8 和上图 2 做个对比，看看 MoCo v3 的训练方法和 MoCo v1/2 的训练方法的差异。

![](https://pic1.zhimg.com/v2-8df3d9d6f4ff498505ea1a75c26c092c_r.jpg)

MoCo v3 的训练方法和 MoCo v1/2 的训练方法的差异是：

* **取消了 Memory Queue 的机制：**你会发现整套 Framework 里面没有 Memory Queue 了，那这意味着什么呢？这就意味着 MoCo v3 所观察的负样本都来自一个 Batch 的图片，也就是图 8 里面的 **n**。换句话讲，只有**当 Batch size 足够大时，模型才能看到足够的负样本。**那么 MoCo v3 具体是取了 **4096** 这样一个巨大的 Batch size。
* **Encoder** $f_\textrm{q}$ 除了 Backbone 和预测头 Projection head 以外，还添加了个 **Prediction head**，是遵循了 BYOL 这篇论文的方法。
* 对于同一张图片的 2 个增强版本 $\color{red}{x_1},\color{darkgreen}{x_2}$ ，分别通过 **Encoder** $f_\textrm{q}$ 和 **Momentum** **Encoder** $f_\textrm{Mk}$ 得到 $\color{red}{q_1},\color{darkgreen}{q_2}$ 和 $\color{red}{k_1},\color{darkgreen}{k_2}$ 。让 $\color{red}{q_1},\color{darkgreen}{k_2}$ 通过 Contrastive loss (式 1) 进行优化 **Encoder** $f_\textrm{q}$ 的参数，让 $\color{darkgreen}{q_2},\color{red}{k_1}$ 通过 Contrastive loss (式 1) 进行优化 **Encoder** $f_\textrm{q}$ 的参数。**Momentum Encoder** $f_\textrm{Mk}$ 通过式 2 进行动量更新。

$$
\theta_\textrm{k}\leftarrow m\theta_\textrm{k}+(1-m)\theta_\textrm{q}\tag{2}
$$

下面是伪代码， $f_\textrm{q}$ 和 $f_\textrm{Mk}$ 在代码里面分别表示为：**f_q** , **f_k。**

**1) 数据增强：**

现在我们有一堆无标签的数据，拿出一个 Batch，代码表示为 **x**，也就是 $N$ 张图片，分别进行两种不同的数据增强，得到 **x_1** 和 **x_2**，则 **x_1** 是 $N$ 张图片，**x_2** 也是 $N$ 张图片。

```
for x in loader: # load a minibatch x with N samples
    x1, x2 = aug(x), aug(x) # augmentation
```

**2) 分别通过 Encoder 和 Momentum Encoder：**

**x_1** 分别通过 Encoder 和 Momentum Encoder 得到特征 **q_1** 和 **k_1**，维度是 $N,C$ ，这里特征空间由一个长度为 $C=128$ 的向量表示。

**x_2** 分别通过 Encoder 和 Momentum Encoder 得到特征 **q_2** 和 **k_2**，维度是 $N,C$ ，这里特征空间由一个长度为 $C=128$ 的向量表示。

```
q1, q2 = f_q(x1), f_q(x2) # queries: [N, C] each
    k1, k2 = f_k(x1), f_k(x2) # keys: [N, C] each
```

**3) 通过一个 Contrastive loss 优化 q_1 和 k_2，通过另一个 Contrastive loss 优化 q_2 和 k_1，并反向传播更新 f_q 的参数：**

```
loss = ctr(q1, k2) + ctr(q2, k1) # symmetrized
    loss.backward()
    update(f_q) # optimizer update: f_q
```

**4) Contrastive loss 的定义：**

**对两个维度是 (N,C) 的矩阵 (比如是 q_1 和 k_2) 做矩阵相乘，得到维度是 (N,N) 的矩阵，其对角线元素代表的就是 positive sample 的相似度，就是让对角线元素越大越好，所以目标是整个这个 (N,N) 的矩阵越接近单位阵越好，如下所示。**

```
def ctr(q, k):
    logits = mm(q, k.t()) # [N, N] pairs
    labels = range(N) # positives are in diagonal
    loss = CrossEntropyLoss(logits/tau, labels)
    return 2 * tau * loss
```

**5) Momentum Encoder 的参数使用动量更新：**

```
f_k = m*f_k + (1-m)*f_q # momentum update: f_k
```

**全部的伪代码 (来自 MoCo v3 的 paper)：**

```
# f_q: encoder: backbone + pred mlp + proj mlp
# f_k: momentum encoder: backbone + pred mlp
# m: momentum coefficient
# tau: temperature
for x in loader: # load a minibatch x with N samples
    x1, x2 = aug(x), aug(x) # augmentation
    q1, q2 = f_q(x1), f_q(x2) # queries: [N, C] each
    k1, k2 = f_k(x1), f_k(x2) # keys: [N, C] each
    loss = ctr(q1, k2) + ctr(q2, k1) # symmetrized
    loss.backward()
    update(f_q) # optimizer update: f_q
    f_k = m*f_k + (1-m)*f_q # momentum update: f_k
# contrastive loss
def ctr(q, k):
    logits = mm(q, k.t()) # [N, N] pairs
    labels = range(N) # positives are in diagonal
    loss = CrossEntropyLoss(logits/tau, labels)
    return 2 * tau * loss
```

以上就是 MoCo v3 的全部方法，都可以概括在图 8 里面。它的性能如何呢？假设 Encoder 依然取 ResNet-50，则 MoCo v2，MoCo v2+，MoCo v3 的对比如下图 9 所示，主要的提点来自于**大的 Batch size (4096)** 和 **Prediction head 的使用**。

![](https://pic1.zhimg.com/v2-dcc7a01b1fc2a83856713aba81ad7200_r.jpg)

* **2.2 MoCo v3 自监督训练 ViT 的不稳定性**

上图 9 的实验结果证明了 MoCo v3 在 **Encoder 依然取 ResNet-50** 时的有效性。那么当 **Encoder 变成 Transformer** 时的情况又如何呢？如本节一开始所述，作者给出的结论是：**影响自监督 ViT 模型训练的关键是：instability，即训练的不稳定性。**而这种训练的不稳定性所造成的结果**并不是训练过程无法收敛 (convergence)，而是性能的轻微下降 (下降 1%-3% 的精度)。**

**Batch size 过大使得训练不稳定**

如下图 10 所示是使用 MoCo v3 方法，Encoder 架构换成 ViT-B/16 ，Learning rate=1e-4，在 ImageNet 数据集上训练 100 epochs 的结果。作者使用了 4 种不同的 Batch size：1024, 2048, 4096, 6144 的结果。可以看到当 bs=4096 时，曲线出现了 dip 现象 (稍稍落下又急速升起)。这种不稳定现象导致了精度出现下降。当 bs=6144 时，曲线的 dip 现象更大了，可能是因为跳出了当前的 local minimum。这种不稳定现象导致了精度出现了更多的下降。

![](https://pic3.zhimg.com/v2-6f340a78cd66e0d0c743b30ab7bc1c6e_r.jpg)

**Learning rate 过大使得训练不稳定**

如下图 11 所示是使用 MoCo v3 方法，Encoder 架构换成 ViT-B/16 ，Batch size=4096，在 ImageNet 数据集上训练 100 epochs 的结果。作者使用了 4 种不同的 Learning rate：0.5e-4, 1.0e-4, 1.5e-4 的结果。可以看到当 Learning rate 较大时，曲线出现了 dip 现象 (稍稍落下又急速升起)。这种不稳定现象导致了精度出现下降。

![](https://pic4.zhimg.com/v2-6050f0c15187391035113a4cbd1e09d7_r.jpg)

**LARS optimizer 的不稳定性**

如下图 12 所示是使用 MoCo v3 方法，Encoder 架构换成 ViT-B/16 ，Batch size=4096，在 ImageNet 数据集上训练 100 epochs 的结果，不同的是使用了 LARS 优化器，分别使用了 4 种不同的 Learning rate：3e-4, 5e-4, 6e-4, 8e-4 的结果。结果发现当给定合适的学习率时，LARS 的性能可以超过 AdamW，但是当学习率稍微变大时，性能就会显著下降。而且曲线自始至终都是平滑的，没有 dip 现象。所以最终为了使得训练对学习率更鲁棒，作者还是采用 AdamW 作为优化器。因为若采用 LARS，则每换一个网络架构就要重新搜索最合适的 Learning rate。

![](https://pic2.zhimg.com/v2-67a70be21e86af66b827b9060016c10d_r.jpg)

* **2.3 提升训练稳定性的方法：冻结第 1 层 (patch embedding 层) 参数**

上面图 10-12 的实验表明 Batch size 或者 learning rate 的细微变化都有可能导致 Self-Supervised ViT 的训练不稳定。作者发现**导致训练出现不稳定的这些 dip 跟梯度暴涨 (spike) 有关**，如下图 13 所示，**第 1 层会先出现梯度暴涨的现象，结果几十次迭代后，会传到到最后 1 层**。

![](https://pic3.zhimg.com/v2-c44f2e0a45a9c641d79291ca0347fbd6_r.jpg)

所以说**问题就出在第 1 层出现了梯度暴涨啊**，一旦第 1 层梯度暴涨，这个现象就会在几十次迭代之内传遍整个网络。所以说想解决训练出现不稳定的问题就不能让第 1 层出现梯度暴涨！

所以作者解决的办法是冻结第 1 层的参数 ，也就是 patch embedding 那层，随机初始化后，不再更新这一层的参数，然后发现好使，如图 14 所示。

patch embedding 那层具体就是一个 $k=p=16,s=p=16$ 的卷积操作，输入 channel 数是 3，输出 channel 数是 embed_dim=768/384/192。

**patch embedding 代码：**

```
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

如下图 14,15 所示是使用 MoCo v3 or SimCLR, BYOL 方法，Encoder 架构换成 ViT-B/16 ，Batch size=4096，在 ImageNet 数据集上训练 100 epochs 的结果，不同的是冻结了 patch embedding 那层的参数，使用了随机参数初始化。

图 14 和 15 表明不论是 MoCo v3 还是 SimCLR, BYOL 方法，冻结 patch embedding 那层的参数都能够提升自监督 ViT 的训练稳定性。除此之外， gradient-clip 也能够帮助提升训练稳定性，其极限情况就是冻结参数。

![](https://pic1.zhimg.com/v2-5ba98a388c75f828377d02433cb5c0dc_r.jpg)

![](https://pic2.zhimg.com/v2-9e0999a94a7bbcd176d81a1abe884f3d_r.jpg)

* **2.4 MoCo v3 实验**

**超参数细节**

<table data-draft-node="block" data-draft-type="table" data-size="normal" data-row-style="normal"><tbody><tr><th>超参数</th><th>具体值</th></tr><tr><td>Optimizer</td><td>AdamW，bs=4096，epochs=100，搜索 lr 和 wd 的最优解，warmup=40 epochs，cosine decay learning rate</td></tr><tr><td>Projection head</td><td>3 层的 MLP，激活函数 ReLU, hidden dim=4096，output dim=256</td></tr><tr><td>Prediction head</td><td>2 层的 MLP，激活函数 ReLU, hidden dim=4096，output dim=256</td></tr><tr><td>loss</td><td>超参数 tau=0.2</td></tr><tr><td>评价指标</td><td>Linear Evaluation (或者叫 Linear Classification，Linear Probing)，在本专栏之前的文章介绍过，具体做法是：冻结 Encoder 的参数，在 Encoder 之后添加一个分类头 (FC 层 + softmax)，使用全部的标签只训练这个分类头的参数，得到的测试集精度就是自监督模型的精度。</td></tr></tbody></table>

如下图 16 所示是 ViT-S/16 和 ViT-B/16 模型在 4 种自监督学习框架下的性能对比。为了确保对比的公平性，lr 和 wd 都经过了搜索。结果显示 MoCo v3 框架具有最优的性能。

![](https://pic1.zhimg.com/v2-3f6b1ca414008a455aedfcda976eb734_r.jpg)

下图 17 展示的是不同的自监督学习框架对 ViT 和 ResNet 模型的偏爱，可以看出 SimCLR 和 MoCo v3 这两个自监督框架在 ViT 类的 Transformer 模型上的效果更好。

![](https://pic1.zhimg.com/v2-38c3839fc14f39207d3cfaeebf899988_b.jpg)

**对比实验：**

**1) 位置编码的具体形式**

如下图 18 所示，最好的位置编码还是余弦编码 sin-cos。在无监督训练过程去除位置编码，效果下降了 1 个多点，说明 ViT 的学习能力很强，在没有位置信息的情况下就可以学习的很好；从另外一个角度来看，也说明 ViT 并没有充分利用好位置信息。

![](https://pic4.zhimg.com/v2-a540b0f0702bc5a0a38e9ffaa1432f9f_r.jpg)

**2) class token 的必要性**

如下图 19 所示，使用 class token 的性能是 76.5，而简单地取消 class token，并换成 Global Average Pooling 会下降到 69.7，这时候最后一层后面有个 LN 层。如果把它也去掉，性能会提升到 76.3。说明 class token 并不是必要的，LN 的选择也很重要。

![](https://pic3.zhimg.com/v2-6c4816789c533e59784ef40947899dfe_r.jpg)

**3) Prediction head 的必要性**

如下图 20 所示，去掉 Prediction head 会使性能稍微下降。

![](https://pic3.zhimg.com/v2-0d1c304eb58621c5df281d6c396c1192_r.jpg)

**4) momentum 超参数的影响**

如下图 21 所示，momentum 超参数取 0.99 是最优的。m=0 就是 Momentum Encoder $f_\textrm{Mk}$ 的参数和 Encoder $f_\textrm{q}$ 的参数完全一致，那就是 SimCLR 的做法了。

![](https://pic1.zhimg.com/v2-422d6d056c1c56461dc91b97f7ad8dc4_r.jpg)

**MoCo v3 与其他模型的性能对比**

Self-supervised Transformer 的性能对比可以有两个方向，一个是跟 Supervised Transformer 对比，另一个是跟 Self-supervised CNN 对比。

第 1 个方向的对比如下图 22 所示。虽然 MoCo v3-VIT-L 参数量 比 VIT-B 大了很多，但 VIT-B 训练的数据集比 ImageNet 大很多。

![](https://pic3.zhimg.com/v2-49dcbe1c47193f13dcc3fd95dbf3e89e_r.jpg)

第 2 个方向的对比如下图 23 所示。作者跟采用了 Big ResNet 的方法进行对比，以 VIT-L 为 backbone 的 MoCo v3 完胜。注意图 23 这个表的每一列表示的是把 MoCo v3 的方法用在每一列对应的模型上的性能 (比如第 2 列就是在 ViT-B 这种模型使用 MoCo v3)。第 1 行就代表直接使用这个模型，第 2 行代表把 ViT 模型里面的 LN 全部换成 BN 的效果 (以 ViT-BN 表示)，第 3 行代表再把 ViT 模型的 patch 大小设置为 7 以获得更长的 sequence (以 ViT-BN/7 表示)，但是这会使计算量变为 6 倍。而且这里没有列出各个模型的参数量，可能存在不公平对比的情况。

![](https://pic3.zhimg.com/v2-c932d63b1b1f6370c8be2c7432f9557a_r.jpg)

**小结**

MoCo v3 的改进如图 8 所示，取消了 Memory Queue 的机制，添加了个 **Prediction head**，且对于同一张图片的 2 个增强版本 $\color{red}{x_1},\color{darkgreen}{x_2}$ ，分别通过 **Encoder** $f_\textrm{q}$ 和 **Momentum** **Encoder** $f_\textrm{Mk}$ 得到 $\color{red}{q_1},\color{darkgreen}{q_2}$ 和 $\color{red}{k_1},\color{darkgreen}{k_2}$ 。让 $\color{red}{q_1},\color{darkgreen}{k_2}$ 通过 Contrastive loss 进行优化 **Encoder** $f_\textrm{q}$ 的参数，让 $\color{darkgreen}{q_2},\color{red}{k_1}$ 通过 Contrastive loss 进行优化 **Encoder** $f_\textrm{q}$ 的参数。

在 Self-supervised 训练 Transformer 的过程中发现了 instablity 的问题，通过冻住 patch embedding 的参数，以治标不治本的形式解决了这个问题，最终 Self-supervised Transformer 可以 beat 掉 Supervised Transformer 和 Self-supervised CNN。
