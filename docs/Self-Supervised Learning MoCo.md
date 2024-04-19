---
url: https://zhuanlan.zhihu.com/p/382763210
title: MoCo
date: 2024-03-07 19:37:54
tag: 
summary:
---
**Self-Supervised Learning**，又称为自监督学习，我们知道一般机器学习分为有监督学习，无监督学习和强化学习。 而 Self-Supervised Learning 是无监督学习里面的一种，主要是希望能够学习到一种**通用的特征表达**用于**下游任务 (Downstream Tasks)**。 其主要的方式就是通过自己监督自己。作为代表作的 kaiming 的 MoCo 引发一波热议， Yann Lecun 也在 AAAI 上讲 Self-Supervised Learning 是未来的大势所趋。所以在这个系列中，我会系统地解读 Self-Supervised Learning 的经典工作。

今天介绍的 MoCo 这个系列的第一版 MoCo v1 就是在 SimCLR 诞生之前的一种比较流行的无监督学习方法，这个系列的前 2 个工作 MoCo v1 和 v2 是针对 CNN 设计的，而 MoCo v3 是针对最近大火的 Transformer 模型设计的，反映了 MoCo 这类方法**对视觉模型的普适性**。MoCo 和 SimCLR 系列方法的共同特点是**简单有效**，关于 SimCLR 的详细解读欢迎参考下面的链接啊：

[科技猛兽：Self-Supervised Learning 超详细解读 (二)：SimCLR 系列](https://zhuanlan.zhihu.com/p/378953015)

总结下 Self-Supervised Learning 的方法，用 4 个英文单词概括一下就是：

**Unsupervised Pre-train, Supervised Fine-tune.**

在预训练阶段我们使用**无标签的数据集 (unlabeled data)**，因为有标签的数据集**很贵**，打标签得要多少人工劳力去标注，那成本是相当高的，所以这玩意太贵。相反，无标签的数据集网上随便到处爬，它**便宜**。在训练模型参数的时候，我们不追求把这个参数用带标签数据从**初始化的一张白纸**给一步训练到位，原因就是数据集太贵。于是 **Self-Supervised Learning** 就想先把参数从 **一张白纸** 训练到 **初步成型**，再从 **初步成型** 训练到 **完全成型**。注意这是 2 个阶段。这个**训练到初步成型的东西**，我们把它叫做 **Visual Representation**。预训练模型的时候，就是模型参数从 **一张白纸** 到 **初步成型** 的这个过程，还是用无标签数据集。等我把模型参数训练个八九不离十，这时候再根据你 **下游任务 (Downstream Tasks)** 的不同去用带标签的数据集把参数训练到 **完全成型**，那这时用的数据集量就不用太多了，因为参数经过了第 1 阶段就已经训练得差不多了。

第 1 个阶段不涉及任何下游任务，就是拿着一堆无标签的数据去预训练，没有特定的任务，这个话用官方语言表达叫做：**in a task-agnostic way**。第 2 个阶段涉及下游任务，就是拿着一堆带标签的数据去在下游任务上 Fine-tune，这个话用官方语言表达叫做：**in a task-specific way**

**以上这些话就是 Self-Supervised Learning 的核心思想**，如下图 1 所示。

![](https://pic1.zhimg.com/v2-4b4accbd1e92a3d7141f60772e06e90c_r.jpg)

* **1 MoCo v1**

**论文名称：Momentum Contrast for Unsupervised Visual Representation Learning**

**论文地址：**

[https://arxiv.org/pdf/1911.05722.pdf](https://arxiv.org/pdf/1911.05722.pdf)

**开源地址：**

[facebookresearch/moco](https://github.com/facebookresearch/moco)

MoCo 系列也遵循这个思想，预训练的 MoCo 模型也会得到 Visual Representation，它们可以通过 Fine-tune 以适应各种各样的下游任务，比如检测和分割等等。MoCo 在 7 个检测 / 语义分割任务（PASCAL VOC, COCO, 其他的数据集）上可以超过他的有监督训练版本。有时会超出很多。这表明在有监督与无监督表示学习上的差距在许多视觉任务中已经变得非常近了。

自监督学习的关键可以概括为两点：Pretext Task，Loss Function，在下面分别介绍。

* **1.1 自监督学习的 Pretext Task**

Pretext Task 是无监督学习领域的一个常见的术语，其中 "Pretext" 翻译过来是 " 幌子，托词，借口 " 的意思。所以 Pretext Task 专指这样一种任务：这种任务并非我们所真正关心的，但是通过完成它们，我们能够学习到一种很好的表示，这种表示对下游任务很重要。

> The term "pretext" implies that the task being solved is not of genuine interest, but is solved only for the true purpose of learning a good data representation.

我这里举几个例子：

**(1) BERT 的 Pretext Task：**在训练 BERT 的时候，我们曾经在预训练时让它作填空的任务，详见：

[科技猛兽：Self-Supervised Learning 超详细解读 (一)：大规模预训练模型 BERT](https://zhuanlan.zhihu.com/p/378360224)

如下图 2 所示，把这段输入文字里面的一部分随机盖住。就是直接用一个 Mask 把要盖住的 token (对中文来说就是一个字) 给 Mask 掉，具体是换成一个**特殊的字符**。接下来把这个盖住的 token 对应位置输出的向量做一个 Linear Transformation，再做 softmax 输出一个分布，这个分布是每一个字的概率。因为这时候 BERT 并不知道被 Mask 住的字是 " 湾 " ，但是我们知道啊，所以损失就是让这个输出和被盖住的 " 湾 " 越接近越好。

![](https://pic4.zhimg.com/v2-fd9c8e1aa8858269634f0f99906c3357_r.jpg)

通过图 2 这种方式训练 BERT，得到的预训练模型在下游任务只要稍微做一点 Fine-tune，效果就会比以往有很大的提升。

所以这里的 **Pretext Task** 就是**填空的任务**，这个任务和下游任务毫不相干，甚至看上去很笨，但是 BERT 就是通过这样的 Pretext Task 学习到了很好的 **Language Representation**，很好地适应了下游任务。

**(2) SimCLR 的 Pretext Task：**在训练 SimCLR 的时候，我们曾经在预训练时试图教模型区分相似和不相似的事物，详见：

[科技猛兽：Self-Supervised Learning 超详细解读 (二)：SimCLR 系列](https://zhuanlan.zhihu.com/p/378953015)

如下图 3 所示，假设现在有 **1 张**任意的图片 $x$ ，叫做 Original Image，先对它做数据增强，得到 2 张增强以后的图片 $x_i,x_j$ 。接下来把增强后的图片 $x_i,x_j$ 输入到 Encoder 里面，注意这 2 个 Encoder 是共享参数的，得到 representation $h_i,h_j$ ，再把 $h_i,h_j$ 继续通过 Projection head 得到 representation $z_i,z_j$ ，这里的 2 个 Projection head 依旧是共享参数的，且其具体的结构表达式是：

$$
z_i=g(h_i)=W^{(2)}\sigma (W^{(1)}h_i)\tag{1}
$$

接下来的目标就是最大化同一张图片得到的 $z_i^1,z_j^1$ ，最小化不同张图片得到的 $z_i^1,z_i^2,z_j^2,z_i^3,z_j^3,z_i^4,z_j^4,z_i^5,z_j^5,...$ 。

![](https://pic3.zhimg.com/v2-aeed95c198c2683dc98c83af0b33c596_r.jpg)

通过图 3 这种方式训练 SimCLR，得到的预训练模型在下游任务只要稍微做一点 Fine-tune，效果就会比以往有很大的提升。

所以这里的 **Pretext Task** 就是**试图教模型区分相似和不相似的事物**，这个任务和下游任务毫不相干，甚至看上去很笨，但是 SimCLR 就是通过这样的 Pretext Task 学习到了很好的 **Image Representation**，很好地适应了下游任务。

还有一些常见的 Pretext Task 诸如 denoising auto-encoders，context autoencoders，cross-channel auto-encoders 等等，这里就不一一介绍了。

* **1.2 自监督学习的 Contrastive loss**

Contrastive loss 来自于下面这篇 Yann LeCun 组的工作，如何理解这个对比损失呢？

[http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

常见的损失函数是 cross entropy loss，它适合于数据的 label 是 one-hot 向量的形式。此时网络结构的最后一层是 softmax，输出得到各个类的预测值。比如现在有 3 个类：dog, cat, horse，它们的 label 分别对应着 (1,0,0), (0,1,0), (0,0,1)，cross entropy loss 会让 dog 图片的输出尽量接近 (1,0,0)，让 cat 图片的输出尽量接近 (0,1,0)，让 horse 图片的输出尽量接近 (0,0,1)。但是这也存在一个问题，就是假设再来 3 个类，分别是 sky，car 和 bus。那么按道理 **dog 与 horse 的距离 应该比 dog 与 sky 的距离近， 因为 dog 与 horse 都属于动物；** car 与 bus 的距离 应该也比 car 与 cat 的距离近，因为 car 与 bus 都属于车类。但是 cross entropy loss 确是一视同仁地把 dog 与 horse 的距离 和 dog 与 sky 的距离 看作是一样的。

**Contrastive loss 的初衷**是想让：1 相近的样本之间的距离越小越好。2 相远的样本之间的距离越大越好。

如果神经网络的损失函数只满足条件 1，那么网络会让任何的输入都输出相同的值，不论输入是 dog, cat, horse 还是 sky，car, bus，输出都是一样的，这确实满足了相近的样本之间的距离越小越好，但是却使得网络丧失了分类能力。

如果神经网络的损失函数只满足条件 1 和 2，是不是就完善了呢？

实际上如果想让相远的样本之间的距离越大越好，就需要一个边界，否则如果 dog 是 (1,0,0)，那么 假设第 1 轮训练网络输出 cat 是 (0,1,0)，第 2 轮训练网络输出 cat 是 (0,5,0)... 这样下去 dog 与 cat 之间的距离越来越大，网络却没法收敛。

**Contrastive loss 改进的思路**就是让相远的样本之间的距离越大越好，但是这个距离要有边界，即要求：**1 相近的样本之间的距离越小越好。2 相远的样本之间的距离越大越好，这个距离最大是** $m$ **。** 如下图 4 弹簧图所示：黑色实心球代表与蓝色球相近的样本，白色空心球代表与蓝色球相远的样本，蓝色箭头的长度代表力的大小，方向代表力的方向。

(a)：Contrastive loss 使得相近的样本接近。

(b)：横轴代表样本之间的距离，纵轴代表 loss 值。Contrastive loss 使得相近的样本距离越小，loss 值越低。

(c)：Contrastive loss 使得相远的样本疏远。

(d)：横轴代表样本之间的距离，纵轴代表 loss 值。Contrastive loss 使得相远的样本距离越大，loss 值越低，但距离存在上界，就是红线与 x 轴的交点 $m$ ，代表距离的最大值。

(e)：一个样本受到其他各个独立样本的作用，各个方向的力处于平衡状态，代表模型参数在 Contrastive loss 的作用下训练到收敛。

![](https://pic4.zhimg.com/v2-71c4b9cb6c99bf6534afc76442d09ac7_r.jpg)

Contrastive loss 用公式表示为：

$$
\begin{align} &L(W,Y,\vec{X_1},\vec{X_2})=(1-Y)\frac{1}{2}(D_W)^2+(Y)\frac{1}{2}\left\{ \max(0,m-D_W) \right\}^2\\ &D_W(\vec{X_1},\vec{X_2})=||G_W(\vec{X_1})-G_W(\vec{X_2})||_2 \end{align} \tag{2}
$$

式中， $\vec{X_1},\vec{X_2}$ 是 2 个样本， $D_W$ 是 $\vec{X_1}$ 与 $\vec{X_2}$ 在潜变量空间的欧几里德距离， $Y=0$ 代表是相近的样本，此时要求 $D_W$ 尽量接近 0。 $Y=1$ 代表是相远的样本，此时要求 $D_W<m$ 时与 $m$ 越接近越好，但是距离一旦超过 $m$ 即失效，不会再接着更新参数使得距离越来越大。

有意思的是那些属于不同类，但两两距离天生大于 m 的样本对。 LeCun 的对比损失完全忽视这些样本对，大大减少了计算量。此外，Contrastive loss 提供了一个不同类别样本点之间距离的下界 $m$ 。

* **1.3 MoCo v1 之前的做法**

了解了 Pretext Task 和 Contrastive loss，接下来就要正式介绍 MoCo v1 的原理了。

![](https://pic2.zhimg.com/v2-1b88c744eb6a339ddfb960d1fc770695_r.jpg)

如上图 5 所示，输入 $x^q$ 通过编码器 Encoder 得到 query $q$ ，还有几个输入 $x^{k_0},x^{k_1},x^{k_2}$ 通过编码器 Encoder 得到 key $k_0,k_1,k_2$ 。假设只有一个 key $k_{+}$ 和 $q$ 是匹配的。根据上面的 Contrastive loss 的性质，只有当 $q$ 和相匹配的 $k_{+}$ 相近，且与其他不匹配的 $k_0,k_1,k_2$ 相远时， Contrastive loss 的值才会最小。这个工作使用点积作为相似度的度量，且使用 InfoNCE 作为 Contrastive loss，则有：

$$
\mathcal{L}_q=-\log \frac{\exp \left(q \cdot k_{+} / \tau\right)}{\sum_{i=0}^K \exp \left(q \cdot k_i / \tau\right)}
$$

式中， $\tau$ 是超参数，这个式子其实非常像把 $q$ 分类成第 $k_{+}$ 类的 cross entropy loss。

这里的 $x^q,x^k$ 可以是 image，可以是 image patches 等等， $q=f_\textrm{q}(x^q),k=f_\textrm{k}(x^k)$ ， $f_\textrm{q},f_\textrm{k}$ 是 Encoder，也可以是很多种架构。

**[1 原始的端到端自监督学习方法]：**对于给定的一个样本 $x^q$ ， 选择一个正样本 $x^{k_{+}}$ (这里正样本的对于图像上的理解就是 $x^q$ 的 data augmentation 版本)。然后选择**一批负样本** (对于图像来说，就是除了 $x^q$ 之外的图像)，然后使用 loss function $\mathcal{L}_q$ 来将 $x^q$ 与正样本之间的距离拉近，负样本之间的距离推开。样本 $x^q$ 输入给 Encoder $f_\textrm{q}$ ，正样本和负样本都输入给 Encoder $f_\textrm{k}$ 。

这样其实就可以做自监督了，就可以进行端到端的训练了。实际上这就是最**原始的图像领域的自监督学习方法**，如下图 6 所示，方法如上面那一段所描述的那样，通过 loss function $\mathcal{L}_q$ 来更新 2 个 Encoder $f_\textrm{q},f_\textrm{k}$ 的参数。

原始的自监督学习方法里面的这**一批负样本**就相当于是有个**字典 (Dictionary)**，字典的 key 就是负样本，字典的 value 就是负样本通过 Encoder $f_\textrm{k}$ 之后的东西。那么现在问题来了：

**问：这一批负样本，即字典的大小是多大呢？**

**答：**负样本的规模就是 batch size，即字典的大小就是 batch size。

举个例子，**假设 batch size = 256**，那么对于给定的一个样本 $x^q$ ， 选择一个正样本 $x^{k_{+}}$ (这里正样本的对于图像上的理解就是 $x^q$ 的 data augmentation 版本)。然后选择 **256 个负样本** (对于图像来说，就是除了 $x^q$ 之外的图像)，然后使用 loss function $\mathcal{L}_q$ 来将 $x^q$ 与正样本之间的距离拉近，负样本之间的距离推开。

毫无疑问是 batch size 越大效果越好的，这一点在 SimCLR 中也得到了证明。但是，由于算力的影响 batch size 不能设置过大，因此很难应用大量的负样本。因此效率较低。

![](https://pic1.zhimg.com/v2-fcce1d06cf3ca759dac77c8b05292d94_r.jpg)

![](https://pic2.zhimg.com/v2-6a069d5cb6d5152513b91dd023cb7e11_b.jpg)

针对很难应用大量的负样本的问题，有没有其他的解决方案呢？下面给出了一种方案，如下图 7 所示。

**[2 采用一个较大的 memory bank 存储较大的字典]：** 对于给定的一个样本 $x^q$ ， 选择一个正样本 $x^{k_{+}}$ (这里正样本的对于图像上的理解就是 $x^q$ 的 data augmentation 版本)。采用一个较大的 memory bank 存储较大的字典，这个 memory bank 具体存储的是所有样本的 representation。(涵盖所有的样本，比如样本一共有 60000 个，那么 memory bank 大小就是 60000，字典大小也是 60000)，采样其中的**一部分负样本** $k^{sample}$ ，然后使用 loss function $\mathcal{L}_q$ 来将 $x^q$ 与正样本之间的距离拉近，负样本之间的距离推开。这次只更新 Encoder $f_\textrm{q}$ 的参数，和几个采样的 key 值 $k^{sample}$ 。因为这时候没有了 Encoder $f_\textrm{k}$ 的反向传播，所以支持 memory bank 容量很大。

但是，你这一个 step 更新的是 Encoder $f_\textrm{q}$ 的参数，和几个采样的 key 值 $\color{purple}{k^{sample_1}}$，下个 step 更新的是 Encoder $f_\textrm{q}$ 的参数，和几个采样的 key 值 $\color{crimson}{k^{sample_2}}$，问题是 $\color{purple}{k^{sample_1}}\ne \color{crimson}{k^{sample_2}}$ ，也就是：Encoder $f_\textrm{q}$ 的参数每个 step 都更新，但是某一个 $k_i$ 可能很多个 step 才被采样到更新一次，而且一个 epoch 只会更新一次。这就出现了一个问题，即：每个 step 编码器都会进行更新，这样最新的 query 采样得到的 key 可能是好多个 step 之前的编码器编码得到的 key，因此丧失了一致性。

![](https://pic1.zhimg.com/v2-05b73c45c1db77ff7022e59499cfb62c_b.jpg)

从这一点来看，[1 原始的端到端自监督学习方法] 的一致性最好，但是受限于 batchsize 的影响。而 [2 采用一个较大的 memory bank 存储较大的字典] 的字典可以设置很大，但是一致性却较差，这看起来似乎是一个不可调和的矛盾。

* **1.4 MoCo v1 的做法**

**[3 MoCo 方法]：**

kaiming 大神利用 momentum (移动平均更新模型权重) 与 queue (字典) 轻松的解决这个问题。为了便于读者理解，这里结合 kaiming 大神提供的伪代码一起讲解 (下面加粗的黑体字母是代码中的变量)。

![](https://pic2.zhimg.com/v2-413229578583cf29b1e592fb513d6399_r.jpg)

![](https://pic3.zhimg.com/v2-35428f2dd8a11823d853b207c871157e_r.jpg)

首先我们假设 Batch size 的大小是 $N$ ，然后现在有个队列 Queue，这个队列的大小是 $K(K>N)$ ，注意这里 $K$ 一般是 $N$ 的数倍，比如 $K=3N,K=5N,...$ ，但是 $K$ 总是比 $N$ 要大的 (代码里面 $K=65536$ ，即队列大小实际是 65536)。

下面如上图 8 所示，有俩网络，一个是 Encoder $f_\textrm{q}$ ，另一个是 Momentum Encoder $f_\textrm{Mk}$ 。这两个模型的网络结构是一样的，初始参数也是一样的 (但是训练开始后两者参数将不再一样了)。 $f_\textrm{q}$ 与 $f_\textrm{Mk}$ 是将输入信息映射到特征空间的网络，特征空间由一个长度为 $C$ 的向量表示，它们在代码里面分别表示为：**f_q** , **f_k** 和 **C**。

代码里的 **k** 可以看作模板，**q** 看作查询元素，每一个输入未知图像的特征由 **f_q** 提取， 现在给一系列由 **f_k** 提取的模板特征 (比如狗的特征、猫的特征) ，就能使用 **f_q** 与 **f_k** 的度量值来确定 **f_q** 是属于什么。

**1) 数据增强：**

现在我们有一堆无标签的数据，拿出一个 Batch，代码表示为 **x**，也就是 $N$ 张图片，分别进行两种不同的数据增强，得到 **x_q** 和 **x_k**，则 **x_q** 是 $N$ 张图片，**x_k** 也是 $N$ 张图片。

```
for x in loader: # 输入一个图像序列x，包含N张图，没有标签
    x_q = aug(x) # 用于查询的图 (数据增强得到)
    x_k = aug(x) # 模板图 (数据增强得到)，自监督就体现在这里，只有图x和x的数据增强才被归为一类
```

**2) 分别通过 Encoder 和 Momentum Encoder：**

**x_q** 通过 Encoder 得到特征 **q**，维度是 $N,C$ ，这里特征空间由一个长度为 $C=128$ 的向量表示。

**x_q** 通过 Momentum Encoder 得到特征 **k**，维度是 $N,C$ 。

```
q = f_q.forward(x_q) # 提取查询特征，输出NxC
    k = f_k.forward(x_k) # 提取模板特征，输出NxC
```

**3) Momentum Encoder 的参数不更新：**

```
# 不使用梯度更新f_k的参数，这是因为文章假设用于提取模板的表示应该是稳定的，不应立即更新
    k = k.detach()
```

**4) 计算 $N$ 张图片的自己与自己的增强图的特征的匹配度：**

```
# 这里bmm是分批矩阵乘法
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1)) # 输出Nx1，也就是自己与自己的增强图的特征的匹配度
```

这里得到的 **l_pos** 的维度是 **(N, 1, 1)**，**N** 个数代表**$N$ 张图片的自己与自己的增强图的特征的匹配度。**

**5) 计算** $N$ **张图片与队列中的** $K$ **张图的特征的匹配度：**

```
l_neg = mm(q.view(N,C), queue.view(C,K)) # 输出Nxk，自己与上一批次所有图的匹配度（全不匹配）
```

这里得到的 **l_neg** 的维度是 **(N, K)**，代表 $N$ **张图片与队列 Queue 中的**$K$ **张图的特征的匹配度。**

**6) 把 4, 5 两步得到的结果 concat 起来：**

```
logits = cat([l_pos, l_neg], dim=1) # 输出Nx(1+k)
```

这里得到的 **logits** 的维度是 **(N, K+1)**，把它看成是一个矩阵的话呢，有 **N** 行，代表一个 Batch 里面的 **N** 张图片。每一行的第 1 个元素是某张图片自己与自己的匹配度，每一行的后面 **K** 个元素是某张图片与其他 **K** 个图片的匹配度，如下图 9 所示，图 9 展示的是某一行的信息，这里的 **K=2**。

![](https://pic3.zhimg.com/v2-f0149860b4586944a8d19dbf1534f6ca_r.jpg)

**7) NCE 损失函数，就是为了保证自己与自己衍生的匹配度输出越大越好，否则越小越好：**

```
labels = zeros(N)
    # NCE损失函数，就是为了保证自己与自己衍生的匹配度输出越大越好，否则越小越好
    loss = CrossEntropyLoss(logits/t, labels) 
    loss.backward()
```

**8) 更新 Encoder 的参数：**

```
update(f_q.params) # f_q使用梯度立即更新
```

**9) Momentum Encoder 的参数使用动量更新：**

```
# 由于假设模板特征的表示方法是稳定的，因此它更新得更慢，这里使用动量法更新，相当于做了个滤波。
    f_k.params = m*f_k.params+(1-m)*f_q.params
```

**10) 更新队列，删除最老的一个 Batch，加入一个新的 Batch：**

```
enqueue(queue, k) # 为了生成反例，所以引入了队列
    dequeue(queue)
```

**全部的伪代码 (来自 MoCo 的 paper)：**

```
f_k.params = f_q.params # 初始化
for x in loader: # 输入一个图像序列x，包含N张图，没有标签
    x_q = aug(x) # 用于查询的图（数据增强得到）
    x_k = aug(x) # 模板图（数据增强得到），自监督就体现在这里，只有图x和x的数据增强才被归为一类
    q = f_q.forward(x_q) # 提取查询特征，输出NxC
    k = f_k.forward(x_k) # 提取模板特征，输出NxC
    # 不使用梯度更新f_k的参数，这是因为文章假设用于提取模板的表示应该是稳定的，不应立即更新
    k = k.detach() 
    # 这里bmm是分批矩阵乘法
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1)) # 输出Nx1，也就是自己与自己的增强图的特征的匹配度
    l_neg = mm(q.view(N,C), queue.view(C,K)) # 输出Nxk，自己与上一批次所有图的匹配度（全不匹配）
    logits = cat([l_pos, l_neg], dim=1) # 输出Nx(1+k)
    labels = zeros(N)
    # NCE损失函数，就是为了保证自己与自己衍生的匹配度输出越大越好，否则越小越好
    loss = CrossEntropyLoss(logits/t, labels) 
    loss.backward()
    update(f_q.params) # f_q使用梯度立即更新
    # 由于假设模板特征的表示方法是稳定的，因此它更新得更慢，这里使用动量法更新，相当于做了个滤波。
    f_k.params = m*f_k.params+(1-m)*f_q.params 
    enqueue(queue, k) # 为了生成反例，所以引入了队列
    dequeue(queue)
```

* **1.5 MoCo v1 FAQ**

以上 10 步就是 MoCo 算法的流程。先把上面的流程搞清楚以后，我们思考以下几个问题：

**1 Encoder $f_\textrm{q}$ 和 Momentum Encoder $f_\textrm{Mk}$ 的输入分别是什么？**

**答：** Encoder $f_\textrm{q}$ 的输入是一个 Batch 的样本 **$x$** 的增强版本 **$x_q$**。Momentum Encoder $f_\textrm{Mk}$ 的输入是一个 Batch 的样本 **$x$** 的另一个增强版本 **$x_k$ 和 队列中的所有样本 $x_{queue}$，$x_{queue}$** 通过 Momentum Encoder 得到代码中的变量 **queue**。

**2** **Encoder** $f_\textrm{q}$ **和 Momentum Encoder** $f_\textrm{Mk}$ **的更新方法有什么不同？**

**答：**Encoder $f_\textrm{q}$ 在每个 step 都会通过**反向传播**更新参数，假设 1 个 epoch 里面有 500 个 step，Encoder $f_\textrm{q}$ 就更新 500 次。Momentum Encoder $f_\textrm{Mk}$ 在每个 step 都会通过动量的方式更新参数，假设 1 个 epoch 里面有 500 个 step，Momentum Encoder $f_\textrm{Mk}$ 就更新 500 次，只是更新的方式是：

$$
\theta_\textrm{k}\leftarrow m\theta_\textrm{k}+(1-m)\theta_\textrm{q}\tag{4}
$$

式中， $m$ 是动量参数，默认取 $m=0.999$ ，这意味着 Momentum Encoder 的更新是极其缓慢的，而且并不是通过反向传播来更新参数，而是通过动量的方式来更新。

**3 MoCo 相对于原来的两种方法的优势在哪里？**

**答：在 [1 原始的端到端自监督学习方法]** 里面，Encoder $f_\textrm{q}$ 和 Encoder $f_\textrm{k}$ 的参数每个 step 都更新，这个问题在前面也有提到，因为 Encoder $f_\textrm{k}$ 输入的是一个 Batch 的 negative samples，所以输入的数量不能太大，即 dictionary 不能太大，即 Batch size 不能太大。

现在的 Momentum Encoder $f_\textrm{Mk}$ 的更新是通过 4 式，以动量的方法更新的，不涉及反向传播，所以 $f_\textrm{Mk}$ 输入的负样本 (negative samples) 的数量可以很多，具体就是 Queue 的大小可以比较大，那当然是负样本的数量越多越好了。这就是 Dictionary as a queue 的含义，即通过动量更新的形式，使得可以包含更多的负样本。而且 Momentum Encoder $f_\textrm{Mk}$ 的更新极其缓慢 (因为 $m=0.999$ 很接近于 1)，所以 Momentum Encoder $f_\textrm{Mk}$ 的更新相当于是看了很多的 Batch，也就是很多负样本。

在 **[2 采用一个较大的 memory bank 存储较大的字典]** 方法里面，所有样本的 representation 都存在 memory bank 里面，根据上文的描述会带来最新的 query 采样得到的 key 可能是好多个 step 之前的编码器编码得到的 key，因此丧失了一致性的问题。但是 MoCo 的每个 step 都会更新 Momentum Encoder，虽然更新缓慢，但是每个 step 都会通过 4 式更新一下 Momentum Encoder，这样 Encoder $f_\textrm{q}$ 和 Momentum Encoder $f_\textrm{Mk}$ 每个 step 都有更新，就解决了一致性的问题。

* **1.6 MoCo v1 实验**

**Encoder 的具体结构**是 ResNet，Contrastive loss 3 式的超参数 $\tau=0.07$ 。

**数据增强的方式**是 (都可以通过 Torchvision 包实现)：

* Randomly resized image + random color jittering
* Random horizontal flip
* Random grayscale conversion

此外，作者还把 BN 替换成了 Shuffling BN，因为 BN 会欺骗 pretext task，轻易找到一种使得 loss 下降很快的方法。

**自监督训练的数据集是：** ImageNet-1M (1280000 训练集，各类别分布均衡) 和 Instagram-1B (1 billion 训练集，各类别分布不均衡)

**优化器：** SGD，weight decay: 0.0001，momentum: 0.9。

**Batch size：** $N=256$

**初始学习率：** 0.03，200 epochs，在第 120 和第 160 epochs 时分别乘以 0.1，结束时是 0.0003。

**实验一：Linear Classification Protocol**

评价一个自监督模型的性能，最关键和最重要的实验莫过于 **Linear Classification Protocol** 了，它也叫做 **Linear Evaluation**，具体做法就是先使用自监督的方法预训练 Encoder，这一过程不使用任何 label。预训练完以后 Encoder 部分的权重也就确定了，这时候把它的权重 freeze 住，同时在 Encoder 的末尾添加 Global Average Pooling 和一个线性分类器 (具体就是一个 FC 层 + softmax 函数)，并在某个数据集上做 Fine-tune，这一过程使用全部的 label。

上述方法 [1 原始的端到端自监督学习方法]，[2 采用一个较大的 memory bank 存储较大的字典]**，**[3 MoCo 方法] 的结果对比如下图 10 所示。

![](https://pic4.zhimg.com/v2-5c8f92f8b3c6c04a9173a5f864096f6f_r.jpg)

图 10 里面的 $K$ ：

* 对于 [3 MoCo 方法] 来讲就是队列 Queue 的大小，也是负样本的数量。
* 对于 [1 原始的端到端自监督学习方法] 是一个 Batch 的大小，那么这种方法的**一个 Batch 有 1 个正样本和 K-1 个负样本**。因为对于给定的一个样本 $x^q$ ， 选择一个正样本 $x^{k_{+}}$ (这里正样本的对于图像上的理解就是 $x^q$ 的 data augmentation 版本)。然后选择**一批负样本** (对于图像来说，就是除了 $x^q$ 之外的图像)，样本 $x^q$ 输入给 Encoder $f_\textrm{q}$ ，正样本和负样本都输入给 Encoder $f_\textrm{k}$ ，所以有 K-1 个负样本。
* 对于 [2 采用一个较大的 memory bank 存储较大的字典] 方法来讲，也是负样本的数量。

我们看到图 10 中的 3 条曲线都是随着 $K$ 的增加而上升的，证明对于每一个样本来讲，正样本的数量都是一个，随着负样本数量的上升，自监督训练的性能会相应提升。我们看图 10 中的黑色线 $K$ 最大取到了 1024，因为这种方法同时使用反向传播更新 Encoder $f_\textrm{q}$ 和 Encoder $f_\textrm{k}$ 的参数，所以 Batch size 的大小受到了显存容量的限制。

同时橙色曲线是最优的，证明了 MoCo 方法的有效性。

**实验二：对比不同动量参数 $m$**

结果如下图 11 所示， $m$ 取 0.999 时性能最好，当 $m=0$ 时，即 **Momentum Encoder** $f_\textrm{Mk}$ 参数 $\theta_\textrm{k}\leftarrow\theta_\textrm{q}$ ，即直接拷贝 **Encoder** $f_\textrm{q}$ 的参数，则训练失败，说明 2 个网络的参数不可以完全一致。

![](https://pic1.zhimg.com/v2-3a26ffdfa676450c24f2c190401006c8_r.jpg)

**实验三：与其他方法对比**

如下图 12 所示，设置 $K=65536,m=0.999$ ，Encoder 架构是 ResNet-50，MoCo 可以达到 60.6% 的准确度，如果 ResNet-50 的宽度设为原来的 4 倍，则精度可以进一步达到 68.6%，比以往方法更占优。

![](https://pic1.zhimg.com/v2-ec1def59c352ba83a25a85152778fccc_r.jpg)

**实验四：下游任务 Fine-tune 结果**

有了预训练好的模型，我们就相当于是已经把参数训练到了初步成型，这时候再根据你 **下游任务 (Downstream Tasks)** 的不同去用带标签的数据集把参数训练到 **完全成型**，那这时用的数据集量就不用太多了，因为参数经过了第 1 阶段就已经训练得差不多了。

本文的下游任务是：PASCAL VOC Object Detection 以及 COCO Object Detection and Segmentation，主要对比的对象是 **ImageNet 预训练模型 (ImageNet supervised pre-training)**，注意这个模型是使用 100% 的 ImageNet 标签训练的。

**PASCAL VOC Object Detection 结果**

**Backbone:** Faster R-CNN: R50-dilated-C5 或者 R50-C4。

**训练数据尺寸：** 训练时 [480, 800]，推理时 800。

**Evaluation data：** 即测试集是 VOC test2007 set。

如下图 13 是在 **trainval07+12 (约 16.5k images)** 数据集上 Fine-tune 之后的结果，当 Backbone 使用 R50-dilated-C5 时，在 ImageNet-1M 上预训练的 MoCo 模型的性能与有监督学习的性能是相似的。在 Instagram-1B 上预训练的 MoCo 模型的性能超过了有监督学习的性能。当 Backbone 使用 R50-dilated-C5 时，在 ImageNet-1M 或者 Instagram-1B 上预训练的 MoCo 模型的性能都超过了有监督学习的性能。

![](https://pic3.zhimg.com/v2-86bcf0af56f31b97b14de92232f8964a_r.jpg)

接着作者又在下游任务上对比了方法 1,2 和 MoCo 的性能，如下图 14 所示。end-to-end 的方法 (上述方法 1) 和 memory bank 方法 (上述方法 2) 的性能都不如 MoCo。

![](https://pic1.zhimg.com/v2-419457ec70a3d90cb45601084d590714_r.jpg)

**COCO Object Detection and Segmentation 结果**

**Backbone:** Mask R-CNN: FPN 或者 C4。

**训练数据尺寸：**训练时 [640, 800]，推理时 800。

**Evaluation data：**即测试集是 val2017。

如下图 15 是在 **train2017 set (约 118k images)** 数据集上 Fine-tune 之后的结果，图 15 (a)(b) 展示的是 backbone 为 R50-FPN 的结果，图 15 (c)(d) 展示的是 backbone 为 R50-C4 的结果。在 2× schedule 的情况下 MoCo 相比于有监督训练来讲更占优。

![](https://pic3.zhimg.com/v2-58cf75e8c51d63745dbd24c6558ad516_r.jpg)

![](https://pic4.zhimg.com/v2-10d9af709d97cae1f53d1f3234d89adf_r.jpg)

![](https://pic3.zhimg.com/v2-20af57a293069c8c4a634f0696b45416_r.jpg)

![](https://pic3.zhimg.com/v2-3e0a863bba3afce082bde21cd17f6202_r.jpg)

* **1.7 MoCo v1 完整代码解读**

**开源地址：**

[facebookresearch/moco](https://github.com/facebookresearch/moco)

MoCo 系列的整套代码力求和 **PyTorch 训练 ImageNet 的官方实现代码 (如下链接)** 的差距尽量小。

[https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py](https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)

**(a) 使用方法：**

**1 自监督训练 (Unsupervised Training)：**

支持多 GPU，DistributedDataParallel 训练，假设 Encoder 是 ResNet-50，在 8 卡训练：

```
python main_moco.py.\ \$
  -a resnet50.\ \$
  --lr 0.03.\ \$
  --batch-size 256.\ \$
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0.\ \$
  [your imagenet-folder with train and val folders]
```

以上代码默认是 MoCo v1 版本，若想跑 v2，则加上命令行参数：
--mlp --moco-t 0.2 --aug-plus --cos

**2 在评估模型时一般使用 Linear Evaluation (Linear Classification)：**

```
python main_lincls.py.\ \$
  -a resnet50.\ \$
  --lr 30.0.\ \$
  --batch-size 256.\ \$
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar.\ \$
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0.\ \$
  [your imagenet-folder with train and val folders]
```

8 NVIDIA V100 GPUs 结果是：

<table data-draft-node="block" data-draft-type="table" data-size="normal" data-row-style="normal"><tbody><tr><th></th><th>pre-train epochs</th><th>pre-train time</th><th>MoCo v1 top-1 acc.</th><th>MoCo v2 top-1 acc.</th></tr><tr><td>ResNet-50</td><td>200</td><td>53 hours</td><td>60.8±0.2</td><td>67.5±0.1</td></tr></tbody></table>

**3 在下游任务 (Detection) 上做 Fine-tune：**

首先安装 Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

把预训练好的 MoCo 模型转化成 detectron2 的格式：

```
python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
```

把数据集按照 detectron2 的文件夹格式要求放在 ./datasets 文件夹下。

开始训练：

```
python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml.\ \$
--num-gpus 8 MODEL.WEIGHTS ./output.pkl
```

**(b) Unsupervised Pre-training 代码解读：**

**1 分布式训练启动**

```
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
```

**mp.spawn 第一个参数是一个函数**，这个函数将执行训练的所有步骤。从这一步开始，python 将建立多个进程，每个进程都会执行 main_worker 函数。 **第二个参数是开启的进程数目。第三个参数是 main_worker 的函数实参。**
然后看 **main_worker** 的定义，特别注意一下。我们送入的两个实参，但实际形参有三个。第一个形参是进程 id 号 (必须要多加一个形参，且放到第一个位置上)。id 号是从 0 到 (总进程数目 - 1) 的。id 为 0 的进程我们就叫做主进程。之所以需要区分进程，因为我们一般打印日志和存权重文件，不会希望每个进程都做一次相同的事情。我们只在主进程完成这个事情就行了 (用 if 判断一下，如下 main_worker 函数所示)。

```
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
```

**2 构造模型**

```
# create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)
```

moco_dim 指的是 feature dimension (default: 128)。
moco_k 指的是 队列 Queue 的长度，默认 65536。
moco_m 指的是 momentum 参数，默认是 0.999。
moco_t 指的是式 3 中的超参数，默认是 0.07。
mlp 指的是 是否使用预测头 Projection head，默认是 True。

moco.builder.MoCo 具体实现如下：

```
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
```

param_k.data.copy_(param_q.data)： $f_\textrm{Mk}$ 参数初始化时直接拷贝 $f_\textrm{q}$ 的参数。
param_k.requires_grad = False：$f_\textrm{Mk}$ 参数不通过反向传播更新。

**3 (4) 式的动量更新：**

```
@torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
```

Momentum Encoder $f_\textrm{Mk}$ 参数的更新方法是： $\theta_\textrm{k}\leftarrow m\theta_\textrm{k}+(1-m)\theta_\textrm{q}$ 。

**4 出队和入队操作：**

```
@torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
```

**5 模型的前向传播：**

```
def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
```

重要的是这几步：
**计算 query 与正样本 key 之间的相似度：**
l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
**计算 query 与负样本 (来自队列 Queue) 之间的相似度：**
l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
**利用一个 Batch 的负样本更新队列：**
self._dequeue_and_enqueue(k)

**6 使用 DistributedDataParallel 包装模型：**

```
model.cuda()
 # DistributedDataParallel will divide and allocate batch_size to all
 # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
```

**7 数据增强：**

```
# Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

# TwoCropsTransform 的定义如下：
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
```

**MoCo v1** 使用 RandomResizedCrop，RandomGrayscale，ColorJitter，RandomHorizontalFlip 这几种数据增强方式，到了 **MoCo v2** 又增加了：
transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5) 这种方式。

**8 损失函数，优化器，sampler 定义：**

```
# define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
```

在 DistributedDataParallel 的模式下应该使用 torch.utils.data.distributed.DistributedSampler 的采样器。

**9 开始训练：**

```
for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
```

**(c) Linear Evaluation 代码解读：**

Linear Evaluation 的代码整体结构和 Unsupervised Pre-training 是一致的，但是在做 Linear Evaluation 时要注意冻结 Encoder 的参数不更新，而只更新最后分类器的参数即可。代码表示如下：

```
# freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()


    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
```

优化器 optimizer 作用的参数为 parameters，它只包含分类器 **fc 的 weight 和 bias** 这两部分。

另外为了确保除了分类器 fc 的 weight 和 bias 的部分，其余所有参数不发生改变，作者使用了这个 sanity_check 函数。这个函数有 **2 个输入：state_dict(Linear Evaluation 进行了一个 Epoch 之后的模型) 和 pretrained_weights (预训练权重的存放文件夹)**。先从 pretrained_weights 的位置导入权重，命名为 state_dict_pre。

接下来通过 assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()) 比较 state_dict 的权重和 pretrained_weights 是不是一致的，即 Encoder 的权重有没有发生变化。如果有变化就会打印 k is changed in linear classifier training。

```
def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):].\ \$
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()),.\ \$
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")
```

代码剩下的部分和 Unsupervised Pre-training 是一致的。

## 总结

MoCo v1 的改进其实可以总结为 2 点：

**(a)** 在 **[1 原始的端到端自监督学习方法]** 里面，Encoder $f_\textrm{q}$ 和 Encoder $f_\textrm{k}$ 的参数每个 step 都更新，这个问题在前面也有提到，因为 Encoder $f_\textrm{k}$ 输入的是一个 Batch 的 negative samples (N-1 个)，所以输入的数量不能太大，即 dictionary 不能太大，即 Batch size 不能太大。

现在的 Momentum Encoder $f_\textrm{Mk}$ 的更新是通过动量的方法更新的，不涉及反向传播，所以 $f_\textrm{Mk}$ 输入的负样本 (negative samples) 的数量可以很多，具体就是 Queue 的大小可以比较大，可以比 mini-batch 大，属于超参数。队列是逐步更新的在每次迭代时，当前 mini-batch 的样本入列，而队列中最老的 mini-batch 样本出列，那当然是负样本的数量越多越好了。这就是 Dictionary as a queue 的含义，即通过动量更新的形式，使得可以包含更多的负样本。而且 Momentum Encoder $f_\textrm{Mk}$ 的更新极其缓慢 (因为 $m=0.999$ 很接近于 1)，所以 Momentum Encoder $f_\textrm{Mk}$ 的更新相当于是看了很多的 Batch，也就是很多负样本。

**(b)** 在 **[2 采用一个较大的 memory bank 存储较大的字典]** 方法里面，所有样本的 representation 都存在 memory bank 里面，根据上文的描述会带来最新的 query 采样得到的 key 可能是好多个 step 之前的编码器编码得到的 key，因此丧失了一致性的问题。但是 MoCo 的每个 step 都会更新 Momentum Encoder，虽然更新缓慢，但是每个 step 都会通过 4 式更新一下 Momentum Encoder，这样 Encoder $f_\textrm{q}$ 和 Momentum Encoder $f_\textrm{Mk}$ 每个 step 都有更新，就解决了一致性的问题。
