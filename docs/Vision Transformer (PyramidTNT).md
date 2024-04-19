---
url: https://zhuanlan.zhihu.com/p/454761367
title: Vision Transformer(Pyramid TNT)
date: 2024-03-07 00:14:43
tags: 
summary: 本系列已授权极市平台，未经允许不得二次转载，如有需要请私信作者。
---

!!! note "Attention"
    原文地址：https://zhuanlan.zhihu.com/p/454761367

Transformer 是 Google 的团队在 2017 年提出的一种 NLP 经典模型，现在比较火热的 Bert 也是基于 Transformer。Transformer 模型使用了 Self-Attention 机制，**不采用** RNN 的**顺序结构**，使得模型**可以并行化训练**，而且能够**拥有全局信息。**

Transformer in Transformer 针对 ViT 处理图片的方式：将输入图片划分成一个个块 (patch) ，然后针对将这些 patch 看成一个序列 (Sequence) 的不完美之处，提出了一种 TNT 架构，它不仅考虑 patch 之间的信息，还考虑每个 patch 的内部信息，使得 Transformer 模型分别对整体和局部信息进行建模，提升性能。

TNT 架构没有使用 PVT 提出的 Transformer 模型金字塔结构，而金字塔结构在大多数 Vision Transformer 和 MLP 模型上都被证明了有很好的建模性能，所以 Pyramid TNT 作为 TNT 的 Extended Version，进一步验证了金字塔结构对于 TNT Backbone 的作用。

## 41 Pyramid TNT：使用金字塔结构改进的 TNT Baseline

**论文名称：PyramidTNT: Improved Transformer-in-Transformer Baselines with Pyramid Architecture**

**TNT 论文地址：**

[https://arxiv.org/pdf/2103.00112.pdf​arxiv.org/pdf/2103.00112.pdf](https://arxiv.org/pdf/2103.00112.pdf)

**Pyramid TNT 论文地址：**

[https://arxiv.org/pdf/2201.00978.pdf​arxiv.org/pdf/2201.00978.pdf](https://arxiv.org/pdf/2201.00978.pdf)

Transformer 需要的是序列 (Sequence) 的输入信号，而我们有的是 image 这种 2D 的输入信号，那**直接把图片分块以后进行 Flatten 操作**是一种很直觉的处理方式。但是，这种 intuitive 的方法能不能够完美地建模图像，因为我们缺少了一部分非常重要的信息，即：**每个 patch 的内部信息**。

TNT 认为，每个输入的内部信息，即每个 patch 的内部信息，没有被 Transformer 所建模。是一个欠考虑的因素。所以 TNT 使得 Transformer 模型既建模那些不同 patch 之间的关系，也要建模每个 patch 内部的关系。

![](https://pic4.zhimg.com/v2-a0385f43a6dd45ce959e8e68e0372653_r.jpg)

第 1 步还是将输入图片划分成 $n$ 个块 (patch)：

$$
\mathcal{X}=[X^1,X^2,\cdots,X^n]\in\mathbb{R}^{n\times p\times p\times 3}\tag{1}
$$

式中 $p$ 是每个块的大小。ViT，DeiT，IPT，SETR，ViT-FRCNN 到这里就把它们输入 Transformer 了，TNT 为了更好地学习图片中 global 和 local 信息的关系，还要再进行一步。 在 TNT 中，作者将 patch 视为表示图像的视觉 "sentence"。每个 patch 进一步分成 $m$ 个子块，即一个 "sentence" 由一系列视觉 "words" 组成。

$$
\begin{equation} X^i\rightarrow [x^{i,1},x^{i,2},\cdots,x^{i,m}], \end{equation} \tag{2}
$$

式中， $x^{i,j}\in\mathbb{R}^{s\times s\times 3}$ 代表第 $i$ 个视觉 "sentence" 的第 $j$ 个视觉 "words"，这一步其实是把每个 patch 通过 PyTorch 的 unfold 操作划分成更小的 patch，之后把这些小 patch 通过线性投影展平，就得到了：

$$
\begin{equation} {Y}^i=[y^{i,1},y^{i,2},\cdots,y^{i,m}], \quad y^{i,j}=\textit{FC}(\textit{Vec}(x^{i,j})), \end{equation} \tag{3}
$$

其中， $y^{i,j}\in\mathbb{R}^{c}$ 是第 $j$ 个视觉 "words" 的 Embedding， $c$ 代表 Embedding dimension。

如下图 1 所示，输入是一个大 patch，输出的黄色大长条是这个 patch 展平以后的 sentence embedding，输出的彩色小长条是这个 patch 划分成更小的 patch 之后再展平以后的 word embedding。

![](https://pic2.zhimg.com/v2-e3289b761f44c4a79af39800d2f2bd7d_r.jpg)

图 2 的操作进行完之后就得到了大 patch 的 sentence embedding 以及小 patch 的 word embedding。接下来把它们送入 Transformer 的 Block 里面建模特征，如下图 2 所示。Transformer 是由很多 TNT Blocks 组成的，每个 TNT Block 包含 2 个 Transformer Block，分别是：

![](https://pic1.zhimg.com/v2-80d654c25d1cd58ebe1c79e7938972c4_r.jpg)

*   Outer block 建模 sentence embedding 之间的 global relationship。
*   Inner block 建模 word embedding 之间的 local structure information。

这两种 Block 对应 2 个不同的数据流，其中 Outer block 的数据流在不同 patch 之间运行，而 Inner block 的数据流在每个 patch 内部运行。

**Inner Transformer：**

定义 $Y_l^i \in\mathbb{R}^{p'\times p'\times c}=\mathbb{R}^{m\times c}$ ，我们把这个值传入 **Inner Transformer** $T_{in}$ ，则有：

$$
\begin{align} {Y'}_l^i &= Y_{l-1}^i+\textit{MSA}(\textit{LN}(Y_{l-1}^i)),\\ Y_l^i &= {Y'}_{l}^i+\textit{MLP}(\textit{LN}({Y'}_{l}^i)). \end{align} \tag{4}
$$

注意正常的 Transformer 的输入应该是 $(b,n,d)$ 的张量，这里 $b$ 代表 batch size， $n$ 代表序列长度， $d$ 代表 hidden dimension。不考虑 batch size 这一维，就是一个 $(n,d)$ 的矩阵，也可以看做是 $n$ 个 $d$ 维向量，那么对于 Inner Transformer $T_{in}$ 来讲，这里的 $d=mc$ 。也就是说，Inner Transformer $T_{in}$ 的输入是 $n$ 个 $mc$ 维的向量。注意这里的 $Y_l^i \in\mathbb{R}^{p'\times p'\times c}=\mathbb{R}^{m\times c}$ 就是这 $n$ 个向量的其中一个。所以 Inner Transformer 的第 $l$ 个 layer 的输出就可以写为：

$$
\mathcal{Y}_l=[Y_l^1,Y_l^2,\cdots,Y_l^n],\mathcal{Y}_l\in\mathbb{R}^{n\times m\times c}\tag{5}
$$

Inner Transformer $T_{in}$ 建模的是更细小的像素级别的 relationship，例如，在一张人脸中，属于眼睛的像素与眼睛的其他像素更相关，而与前额像素的 relationship 较少。

**Outer Transformer：**

Outer Transformer $T_{out}$ 就相当于是 ViT 中的 Transformer，它建模的是更答大的 patch 级别的 relationship，输入的 patch embedding 使用 ViT 类似的做法，添加 $\color{purple}{\text{class token}}\;Z_\text{class}$ ，它们初始化为 0。

$$
\mathcal{Z}_0=[Z_\text{class},Z_0^1,Z_0^2,\cdots,Z_0^n]\in\mathbb{R}^{(n+1)\times d}\tag{6}
$$

  
定义 $\mathcal{Z}_l^i\in\mathbb{R}^{d}$ 为第 $l$ 个 layer 的第 $i$ 个向量，则 Outer Transformer 的表达式为：

$$
\begin{align} \mathcal{Z'}_l^i &= \mathcal{Z}_{l-1}^i+\textit{MSA}(\textit{LN}(\mathcal{Z}_{l-1}^i)),\\ \mathcal{Z}_l^i &= \mathcal{Z'}_{l}^i+\textit{MLP}(\textit{LN}(\mathcal{Z'}_{l}^i)). \end{align} \tag{7}
$$

那么现在既有 Outer Transformer 的第 $l$ 个 layer 的输出向量：

$$
\mathcal{Z}_l=[Z_\text{class},Z_l^1,Z_l^2,\cdots,Z_l^n]\in\mathbb{R}^{(n+1)\times d}\tag{8}
$$

也有 Inner Transformer 的第 $l$ 个 layer 的输出向量：

$$
\mathcal{Y}_l=[Y_l^1,Y_l^2,\cdots,Y_l^n],\mathcal{Y}_l\in\mathbb{R}^{n\times m\times c}\tag{9}
$$

下面的问题是：要如何把它们结合起来，以融合 global 和 local 的信息呢？

作者采用的方式是：

$$
\begin{equation} Z_{l-1}^i = Z_{l-1}^i + \textit{Vec}(Y_{l-1}^i)W_{l-1}+b_{l-1}, \end{equation} \tag{10}
$$

式中， $Z_{l-1}^i\in\mathbb{R}^{d},\textit{Vec}(\cdot)$ 代表 Flatten 操作， $W_{l-1}\in\mathbb{R}^{mc\times d},b_{l-1}\in\mathbb{R}^{d}$ 代表权重。

通过这种方式，把第 $l$ 个 layer 的第 $i$ 个 sentence embedding 向量和第 $i$ 个 word embedding 向量融合起来，即对应图 2 的结构。

总的来说，TNT Block 第 $l$ 个 layer 的输入和输出可以表示为：

$$
\begin{equation} \mathcal{Y}_l,\mathcal{Z}_l = \textit{TNT}(\mathcal{Y}_{l-1},\mathcal{Z}_{l-1}) \end{equation} \tag{11}
$$

在 TNT Block 中，Inner Transformer 建模 word embedding 之间的 local structure information 之间的关系，而 Outer block 建模 sentence embedding 之间的 global relationship。通过将 TNT Block 堆叠 $L$ 次，作者构建了 Transformer in Transformer。最后，使用一个分类头对图像进行分类。

**位置编码：**

位置编码的作用是让像素间保持空间位置关系，对于图像就是保持二维信息，它对于图像识别任务来讲很重要。具体来说，就需要对 sentence embedding 和 word embedding 分别设计一种位置编码。

*   sentence positional encoding：

作者这里使用的是可学习的 1D 位置编码：

$$
\begin{align} \mathcal{Z}_0 \leftarrow \mathcal{Z}_0 + E_{sentence}, \end{align} \tag{12}
$$

式中， $E_{sentence}\in\mathbb{R}^{(n+1)\times d}$ 是给 sentence embedding 使用的位置编码，它用来编码全局空间信息 (global spatial information)。

*   word positional encoding：

作者这里使用的是可学习的 1D 位置编码：

$$
\begin{align} Y_0^i \leftarrow Y_0^i + E_{word},~ i=1,2,\cdots,n \end{align} \tag{13}
$$

式中， $E_{word}\in\mathbb{R}^{m\times c}$ 是给 word embedding 使用的位置编码，它们用来编码局部相对信息 (local relative information)。

*   **40.2 Pyramid TNT 原理分析：**

TNT 作为一种通用的视觉任务 Backbone，取得了优异的性能。Pyramid TNT 受到 Transformer 模型两种主流改进方法：**金字塔架构 (PVT，Swin Transformer，CycleMLP 等等)** 和**卷积 stem (Convolutional Stem)** 的启发，改进了 TNT 架构。

Pyramid TNT 将它们融入 TNT 中，金字塔架构 (Pyramid Structure) 用于**提取多尺度信息**，卷积 stem (Convolutional Stem) 用于**改善图片分块的方法和使得训练过程更加稳定**。此外，Pyramid TNT 还包括其他一些 trick 比如相对位置编码等。

![](https://pic4.zhimg.com/v2-6991f91c12d2ba4e6b6992f4c1d93223_r.jpg)

**Convolutional Stem**

给定输入图片 $X\in \mathbb{R}^{H\times W}$ ，ViT 的做法是通过一个 $stride=kernel=\text{patch size}$ 的卷积进行图片的分块操作。**Early convolutions help transformers see better (NeurIPS 2021)** 这篇论文发现：将这个卷积操作替换成几个连续的卷积操作能够使得 Transformer 模型获得更好的性能，且对优化器更加鲁棒，增加了优化稳定性。基于这个发现，作者也对 TNT 模型应用了 Convolutional Stem。

具体而言 Pyramid TNT 的 Convolutional Stem 是 5 个 3×3 卷积。对于 Outer Transformer，Convolutional Stem 将输入图片变成 $Y\in \mathbb{R}^{\frac{H}{2}\times \frac{W}{2}\times C}$ ，式中 $C$ 是 sentence embedding 维度。对于 Inner Transformer，Convolutional Stem 将输入图片变成 $Y\in \mathbb{R}^{\frac{H}{8}\times \frac{W}{8}\times D}$ ，式中 $D$ 是 word embedding 的维度。对于位置编码，sentence positional encoding 和 word positional encoding 被分别添加在了 sentence embedding 和 word embedding 上。

**Pyramid Architecture**

**[原始的 TNT 网络]：**

在原始 TNT 中，在每个 Block 中保持相同数量的 tokens，遵循 ViT 的设计方式。视觉 "sentence" 和视觉 "words" 的数量自下而上一直保持不变。

视觉 "sentence" 的特征图分辨率自下而上一直是 $\frac{H}{p}\times\frac{W}{p}=\frac{H}{16}\times\frac{W}{16}=14\times14$ 。

视觉 "words" 的特征图分辨率自下而上一直是 $\frac{p}{4}\times\frac{p}{4}=4\times4$ 。

**[Pyramid TNT 网络]：**

在 Pyramid TNT 中，网络在每个 stage 中保持不同数量的 tokens，遵循 PVT 的设计方式。视觉 "sentence" 和视觉 "words" 的数量自下而上分阶段变化。

视觉 "words" 的特征图分辨率在 4 个 stage 中分别是： $\color{crimson}{\frac{H}{2}\times\frac{W}{2}},\frac{H}{4}\times\frac{W}{4},\frac{H}{8}\times\frac{W}{8},\frac{H}{16}\times\frac{W}{16}$ 。

视觉 "sentence" 的特征图分辨率在 4 个 stage 中分别是： $\color{purple}{\frac{H}{8}\times\frac{W}{8}},\frac{H}{16}\times\frac{W}{16},\frac{H}{32}\times\frac{W}{32},\frac{H}{64}\times\frac{W}{64}$ 。

通过 Convolution Stem，把 **224×224 的输入图片**分成 **8×8 的大 patch**，一共是 28×28 个。所以 Outer Transformer 特征图的分辨率是：$H_{out}\times W_{out}=28\times28=\color{purple}{\frac{H}{8}\times\frac{W}{8}}$ 。

通过 Convolution Stem，把 **8×8 的大 patch** 分成 **2×2 的小 patch**，一共是 4×4×28×28 个。所以 Inner Transformer 特征图的分辨率是：$H_{in}\times W_{in}\times H_{out}\times W_{out}=4\times4\times28\times28=\color{crimson}{\frac{H}{2}\times\frac{W}{2}}$ 。

不同 stage 之间通过一个 $stride=2$ 的卷积操作降低特征分辨率。注意不同的 stage 的 **Outer Transformer**，即**视觉 "sentence" 的特征图分辨率** $\color{purple}{H_{out}\times W_{out}}$ 是大小一直变化的，而不同的 stage 的 **Inner Transformer**，**视觉 "words" 的特征图分辨率**一直是 $\color{crimson}{4\times4}\times \color{purple}{H_{out}\times W_{out}}$ 。

**实验结果**

**分类任务实验结果**

数据集：ImageNet-1k (1,280,000 Training data, 50,000 validation data，1000 classes)

超参数设置：

![](https://pic4.zhimg.com/v2-de291717999802499b427df72528ad0b_r.jpg)

实验结果如下图 5 所示。与原始 TNT 相比，Pyramid TNT 获得了更好的性能。 Pyramid TNT-S 比 TNT-S 少 1.9B 计算量，精度提高了 0.5%。作者还将 Pyramid TNT 与其他有代表性的 CNN、MLP 和基于 Transformer 的模型进行了比较。从结果中，我们可以看到 Pyramid TNT 是最先进的视觉 Backbone。

![](https://pic1.zhimg.com/v2-0d1f4d5ccee3e2ea572cf5549f3c0548_r.jpg)

**目标检测实验结果**

**数据集：**COCO 2017 (118,000 Training data, 50,000 validation data)

**对比的框架：**RetinaNet，Mask R-CNN

**超参数：**Batch size=2，AdamW Optimizer，initial lr=1e-4，在第 8 和第 11 个 Epoch 分别乘以 0.1，weight decay=0.05，"1x" schedule (12 epochs)，输入图片 resize 成 (1333, 800)。

金字塔的四个阶段的空间分辨率被设置为： $\frac{H}{8}\times\frac{W}{8},\frac{H}{16}\times\frac{W}{16},\frac{H}{32}\times\frac{W}{32},\frac{H}{64}\times\frac{W}{64}$ 。作者使用了 $stride=2$ 的转置卷积和 BN 和 GeLU 激活函数加上 $stride=1,kernel=3$ 的卷积和 BN 和 GeLU 激活函数，以产生 $\frac{H}{4}\times\frac{W}{4},\frac{H}{8}\times\frac{W}{8},\frac{H}{16}\times\frac{W}{16},\frac{H}{32}\times\frac{W}{32}$ 的分辨率的特征图。

![](https://pic3.zhimg.com/v2-1becb511c1cdc53d9beeb5a4f1d972ee_r.jpg)

在具有相似计算成本的 one-stage 和 two-stage 的检测器上，Pyramid-S 明显优于其他 Backbone。例如，基于 Pyramid-S 的 RetinaNet 达到了 42.0 $AP$ 和 57.7 $AP_L$ 。这些结果表明，金字塔结构有助于捕获更好的全局信息。  

**实例分割实验结果**

**数据集：**COCO 2017 (118,000 Training data, 50,000 validation data)

**对比的框架：**Mask R-CNN，Cascade Mask R-CNN

**超参数：**Batch size=16，AdamW Optimizer，initial lr=1e-4，在第 27 和第 33 个 Epoch 分别乘以 0.1，weight decay=0.05，"3x" schedule，输入图片 resize 成 (1333, 800)。

![](https://pic1.zhimg.com/v2-9e5588ab4adb0a2b79ea563cf34077dc_r.jpg)

Pyramid-S 在 Mask R-CNN 和 Cascade Mask R-CNN 上可以获得比其他 Backbone 好得多的 $AP^b$ 和 $AP^m$ ，显示出其更好的特征表示能力。例如，Pyramid-S 在 Mask R-CNN 上 Wave-MLP 高出 0.9 的 $AP^b$ 。

*   **40.3 Pyramid TNT 代码解读：**

**代码来自：**

[https://github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch​github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch](https://github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch)

一些张量的维度的大小已经在代码中以注释的形式进行标注。

**Convolutional Stem：**

```python
class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_chans=3, outer_dim=768, inner_dim=24):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.inner_dim = inner_dim
        self.num_patches = img_size[0] // 8 * img_size[1] // 8
        self.num_words = 16
        
        self.common_conv = nn.Sequential(
            nn.Conv2d(in_chans, inner_dim*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*2),
            nn.ReLU(inplace=True),
        )
        self.inner_convs = nn.Sequential(
            nn.Conv2d(inner_dim*2, inner_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
        )
        self.outer_convs = nn.Sequential(
            nn.Conv2d(inner_dim*2, inner_dim*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim*4, inner_dim*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim*8, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(inplace=False),
        )
        
        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)

    def forward(self, x):
        B, C, H, W = x.shape
        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4
        x = self.common_conv(x)
        # inner_tokens
        # inner_tokens: (B, inner_dim, H/2, W/2)
        inner_tokens = self.inner_convs(x) # B, C, H, W
        # inner_tokens: (B, H/8, W/8, inner_dim*4*4)
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2) # B, N, Ck2
        # inner_tokens: (B, inner_dim, H/8*W/8, 4*4)
        inner_tokens = inner_tokens.reshape(B * H_out * W_out, self.inner_dim, H_in*W_in).transpose(1, 2) # B*N, C, 4*4
        # outer_tokens
        # outer_tokens: (B, outer_dim, H/8, W/8)
        outer_tokens = self.outer_convs(x) # B, C, H_out, W_out
        # outer_tokens: (B, H/8*W/8, outer_dim)
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(B, H_out * W_out, -1)
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)
```

注意 Convolution Stem 返回的 inner_tokens 和 outer_tokens 张量的维度：  
outer_tokens: (B, H/8*W/8, outer_dim)  
inner_tokens: (B, inner_dim, H/8*W/8, 4*4)  
Convolution Stem 返回的 inner_tokens 和 outer_tokens 分别通过后面 Block 类的 Inner Attention 和 Outer Attention，二者输出的张量维度分别是：(B*H/8*W/8, 4*4, inner_dim) 和 (B, H/8*W/8, outer_dim)。之后，这两个张量再按照上式 10 中的方式融合在一起。  
其实 Pyramid Transformer in Transformer 代码的核心是通过这个 Convolution Stem 分别得到两个不同维度的张量，一个输入 Outer Transformer Block，一个输入 Inner Transformer Block。这两个 Transformer Block 的输出再按照上式 10 中的方式融合在一起。

**MLP：**

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

**Attention 类 (这里作者用了 PVT V2 的轻量 attention 类的实现)：**

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.pool = nn.AvgPool2d(sr_ratio, stride=sr_ratio)
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, relative_pos=None):
        B, N, C = x.shape
        # q: (B, nH, N, C/nH)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # x_: (B, C, H, W)
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # x_: (B, N/4, C)
            x_ = self.pool(x_).reshape(B, C, -1).permute(0, 2, 1)
            # x_: (B, N/4, C)
            x_ = self.norm(self.linear(x_))
            # x_: (2, B, nH, N/4, C/nH)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k,v: (B, nH, N/4, C/nH)
        k, v = kv[0], kv[1]

        # attn: (B, nH, N, N/4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if relative_pos is not None:
            attn += relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # X: (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

**一个 Pyramid TNT Block 的实现：**

```python
class Block(nn.Module):
    """ TNT Block
    """
    def __init__(self, outer_dim, inner_dim, outer_head, inner_head, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer(num_words * inner_dim)
            self.inner_attn = Attention(
                inner_dim, num_heads=inner_head, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer(num_words * inner_dim)
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)
        # Outer
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, num_heads=outer_head, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)

    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in, relative_pos):
        # outer_tokens: (B, H/8*W/8, outer_dim)
        B, N, C = outer_tokens.size()
        if self.has_inner:
            # x: (B*H/8*W/8, 4*4, inner_dim)
            x = x + self.drop_path(self.inner_attn(self.inner_norm1(x.reshape(B, N, -1)).reshape(B*N, H_in*W_in, -1), H_in, W_in)) # B*N, k*k, c
            # x: (B*H/8*W/8, 4*4, inner_dim)
            x = x + self.drop_path(self.inner_mlp(self.inner_norm2(x.reshape(B, N, -1)).reshape(B*N, H_in*W_in, -1))) # B*N, k*k, c
            # outer_tokens: (B, H/8*W/8, outer_dim)
            outer_tokens = outer_tokens + self.proj_norm2(self.proj(self.proj_norm1(x.reshape(B, N, -1)))) # B, N, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), H_out, W_out, relative_pos))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            # outer_tokens: (B, H/8*W/8, outer_dim)
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), H_out, W_out, relative_pos))
            # outer_tokens: (B, H/8*W/8, outer_dim)
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        # x: (B*H/8*W/8, 4*4, inner_dim)
        # outer_tokens: (B, H/8*W/8, outer_dim)
        return x, outer_tokens
```

和 TNT 基本一致，不同之处是前向函数中还需要传入 H_out, W_out, H_in, W_in, relative_pos 这些参数，它们分别代表大 patch 和小 patch 的特征分辨率大小。

**一个 Pyramid TNT Stage 的实现：**

```python
class Stage(nn.Module):
    """ PyramidTNT stage
    """
    def __init__(self, num_blocks, outer_dim, inner_dim, outer_head, inner_head, num_patches, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks
        
        for j in range(num_blocks):
            if j == 0:
                _inner_dim = inner_dim
            elif j == 1 and num_blocks > 6:
                _inner_dim = inner_dim
            else:
                _inner_dim = -1
            blocks.append(Block(
                outer_dim, _inner_dim, outer_head=outer_head, inner_head=inner_head,
                num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[j], act_layer=act_layer, norm_layer=norm_layer,
                se=se, sr_ratio=sr_ratio))

        self.blocks = nn.ModuleList(blocks)
        self.relative_pos = nn.Parameter(torch.randn(
                        1, outer_head, num_patches, num_patches // sr_ratio // sr_ratio))

    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in, self.relative_pos)
        return inner_tokens, outer_tokens
```

**不同的 stage 之间应有下采样的操作。**"sentence" level 和 "word" level 的下采样分别通过下面的 SentenceAggregation 类和 WordAggregation 类来解决：

```python
class SentenceAggregation(nn.Module):
    """ Sentence Aggregation
    """
    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),
        )
        
    def forward(self, x, H, W):
        B, N, C = x.shape # B, N, C
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        H, W = math.ceil(H / self.stride), math.ceil(W / self.stride)
        x = x.reshape(B, -1, H * W).transpose(1, 2)
        return x, H, W

class WordAggregation(nn.Module):
    """ Word Aggregation
    """
    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),
        )

    def forward(self, x, H_out, W_out, H_in, W_in):
        B_N, M, C = x.shape # B*N, M, C
        x = self.norm(x)
        x = x.reshape(-1, H_out, W_out, H_in, W_in, C)
        
        # padding to fit (1333, 800) in detection.
        pad_input = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 3, 4, 5, 1, 2), (0, W_out % 2, 0, H_out % 2))
            x = x.permute(0, 4, 5, 1, 2, 3)            
        # patch merge
        x1 = x[:, 0::2, 0::2, :, :, :]  # B, H/2, W/2, H_in, W_in, C
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]
        x = torch.cat([torch.cat([x1, x2], 3), torch.cat([x3, x4], 3)], 4) # B, H/2, W/2, 2*H_in, 2*W_in, C
        x = x.reshape(-1, 2*H_in, 2*W_in, C).permute(0, 3, 1, 2) # B_N/4, C, 2*H_in, 2*W_in
        x = self.conv(x)  # B_N/4, C, H_in, W_in
        x = x.reshape(-1, self.dim_out, M).transpose(1, 2)
        return x
```

我们可以发现 "sentence" level 和 "word" level 的下采样都是通过一个卷积操作完成。  
**第 1 个 stage 结束后的下采样：**  
**SentenceAggregation 输入维度：** $(B, \frac{H}{8}\times\frac{W}{8}, \text{outer dim})$，**输出维度：** $(B, \frac{H}{16}\times\frac{W}{16}, 2\times\text{outer dim})$ 。  
**WordAggregation 输入维度：**$(B\times\frac{H}{8}\times\frac{W}{8}, 4\times4,\text{inner dim})$，**输出维度：** $(B\times\frac{H}{16}\times\frac{W}{16}, 4\times4,2\times\text{inner dim})$ 。  
**第 2 个 stage 结束后的下采样：**  
**SentenceAggregation 输入维度：** $(B, \frac{H}{16}\times\frac{W}{16}, 2\times\text{outer dim})$，**输出维度：** $(B, \frac{H}{32}\times\frac{W}{32}, 4\times\text{outer dim})$ 。  
**WordAggregation 输入维度：**$(B\times\frac{H}{16}\times\frac{W}{16}, 4\times4,2\times\text{inner dim})$，**输出维度：** $(B\times\frac{H}{32}\times\frac{W}{32}, 4\times4,4\times\text{inner dim})$ 。  
**第 3 个 stage 结束后的下采样：**  
**SentenceAggregation 输入维度：** $(B, \frac{H}{32}\times\frac{W}{32}, 4\times\text{outer dim})$，**输出维度：** $(B, \frac{H}{64}\times\frac{W}{64}, 4\times\text{outer dim})$ 。  
**WordAggregation 输入维度：**$(B\times\frac{H}{32}\times\frac{W}{32}, 4\times4,4\times\text{inner dim})$，**输出维度：** $(B\times\frac{H}{64}\times\frac{W}{64}, 4\times4,4\times\text{inner dim})$ 。

**Pyramid TNT 整体模型架构：**

```python
class PyramidTNT(nn.Module):
    """ PyramidTNT (Transformer in Transformer) for computer vision
    """
    def __init__(self, configs=None, img_size=224, in_chans=3, num_classes=1000, mlp_ratio=4., qkv_bias=False,
                qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.num_classes = num_classes
        depths = configs['depths']
        outer_dims = configs['outer_dims']
        inner_dims = configs['inner_dims']
        outer_heads = configs['outer_heads']
        inner_heads = configs['inner_heads']
        sr_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 
        self.num_features = outer_dims[-1]  # num_features for consistency with other models 

        self.patch_embed = Stem(
            img_size=img_size, in_chans=in_chans, outer_dim=outer_dims[0], inner_dim=inner_dims[0])
        num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches, outer_dims[0]))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dims[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        depth = 0
        self.word_merges = nn.ModuleList([])
        self.sentence_merges = nn.ModuleList([])
        self.stages = nn.ModuleList([])
        for i in range(4):
            if i > 0:
                self.word_merges.append(WordAggregation(inner_dims[i-1], inner_dims[i], stride=2))
                self.sentence_merges.append(SentenceAggregation(outer_dims[i-1], outer_dims[i], stride=2))
            self.stages.append(Stage(depths[i], outer_dim=outer_dims[i], inner_dim=inner_dims[i],
                        outer_head=outer_heads[i], inner_head=inner_heads[i],
                        num_patches=num_patches // (2 ** i) // (2 ** i), num_words=num_words, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[depth:depth+depths[i]], norm_layer=norm_layer, se=se, sr_ratio=sr_ratios[i])
            )
            depth += depths[i]
        
        self.norm = norm_layer(outer_dims[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(outer_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(outer_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.outer_pos, std=.02)
        trunc_normal_(self.inner_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

 @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in) = self.patch_embed(x)
        inner_tokens += self.inner_pos # B*N, 8*8, C
        outer_tokens += self.pos_drop(self.outer_pos)  # B, N, D
        
        for i in range(4):
            if i > 0:
                inner_tokens = self.word_merges[i-1](inner_tokens, H_out, W_out, H_in, W_in)
                outer_tokens, H_out, W_out = self.sentence_merges[i-1](outer_tokens, H_out, W_out)
            inner_tokens, outer_tokens = self.stages[i](inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        
        outer_tokens = self.norm(outer_tokens)
        return outer_tokens.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```

**不同大小的 Pyramid TNT 配置信息：**

```python
@register_model
def ptnt_ti_patch16_192(pretrained=False, **kwargs):
    outer_dim = 80
    inner_dim = 5
    outer_head = 2
    inner_head = 1
    configs = {
        'depths': [2, 6, 3, 2],
        'outer_dims': [outer_dim, outer_dim*2, outer_dim*4, outer_dim*4],
        'inner_dims': [inner_dim, inner_dim*2, inner_dim*4, inner_dim*4],
        'outer_heads': [outer_head, outer_head*2, outer_head*4, outer_head*4],
        'inner_heads': [inner_head, inner_head*2, inner_head*4, inner_head*4],
    }
    
    model = PyramidTNT(configs=configs, img_size=192, qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_patch16_192']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
```

**小结**

本文介绍了 Pyramid TNT 架构的原理和 PyTorch 代码实现。TNT 作为一种通用的视觉任务 Backbone，取得了优异的性能。Pyramid TNT 受到 Transformer 模型两种主流改进方法：金字塔架构 (PVT，Swin Transformer，CycleMLP 等等) 和卷积 stem (Convolutional Stem) 的启发，改进了 TNT 架构。Pyramid TNT 将它们融入 TNT 中，金字塔架构 (Pyramid Structure) 用于提取多尺度信息，卷积 stem (Convolutional Stem) 用于改善图片分块的方法和使得训练过程更加稳定。此外，Pyramid TNT 还包括其他一些 trick 比如相对位置编码等。