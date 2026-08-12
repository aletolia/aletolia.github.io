# 当模型操纵流形：计数任务的几何

*Sep 15, 2025 · 原文: https://transformer-circuits.pub/2025/linebreaks/index.html*

---

#### 引言

智能系统需要感知能力来理解、预测并驾驭它们所处的环境。这些感官能力反映了在特定环境中生存所需的技能：蝙蝠使用回声定位，候鸟能感知磁场，北极驯鹿随季节调整它们的紫外视觉。但当你的世界由文本构成时，你又能"看见"什么？语言模型会遇到许多受益于视觉或空间推理的基于文本的任务：解析 ASCII 艺术、解读表格，或处理文本换行约束。然而它们唯一的"感官"输入，只是一串表示词元（token）的整数。它们必须从零开始学习感知能力，并在此过程中发展出专门的机制。

在本工作中，我们研究使 Claude 3.5 Haiku 能够执行一项自然感知任务的机制——该任务在预训练语料中很常见，涉及跟踪文档中的位置。我们发现了习得的位置表征，它们在某些方面与执行类似任务的哺乳动物体内的生物神经元（小鼠的"位置细胞"（place cells）和"边界细胞"（boundary cells））颇为相似，但在另一些方面又为语言模型残差流的约束所独有。我们研究了这些表征，发现它们具有双重解释：既可以理解为一族离散特征，也可以理解为一维"特征流形"/"多维特征" \cite{olah2023manifold,olah2024multidimensional,gorton2024curve,engels2025not}。[^1] 在第一种解释中，位置由哪些特征激活及其激活强度决定；在后一种解释中，位置由特征流形上的角度移动决定。类似地，计算本身也具有双重解释：离散电路，或几何变换。

我们研究的任务是等宽文本中的换行（linebreaking）。当训练语料包含源代码、聊天记录、电子邮件存档、扫描文章或有行宽约束的司法裁决时，模型是如何学会预测何时应断行的呢？[^2] 人类的视觉感知让我们几乎完全下意识地完成这件事——写生日贺卡时，你能看出某一行已经写不下、需要另起一行——但语言模型看到的只是一串整数。为了正确预测下一个词元，模型除了要选对下一个词之外，还必须以某种方式数清当前行的字符数，用文档的行宽约束减去该数值，再把剩余字符数与下一个词的长度作比较。举个具体例子，考虑下面这对提示词，它们隐含着 50 字符的行宽约束。[^3] 当下一个词放得下时，模型就输出它；放不下时，模型就断行：

![](figures/linebreaks/fig-01.png)

为了摸清计算的各个阶段，我们先用离散的字典特征来研究这个模型。在这一框架下，我们可以把计算理解为一张"归因图" \cite{ameisen2025circuit}：一连串特征级联地相互激发或抑制。[^4]

![Claude 3.5 Haiku 在铝提示词中预测换行符的归因图。我们看到与"上一行的宽度"和"当前行中的位置"相关的特征，它们共同激活了"距行限的距离"特征；再结合对计划中下一个词的编码特征，这些特征进而激活"预测换行"特征。](figures/linebreaks/fig-02.png)

归因图展示了模型如何通过组合代表其需要跟踪的不同概念的特征来执行这一任务：

1. 模型通过累加各词元长度的特征，计算出当前行内位置（字符数）的特征，以及整行行宽（约束）的特征。
1. 随后，模型将这两种表征——当前位置与行宽——结合起来，估算距行尾的距离，从而得到"剩余字符数"特征。
1. 最后，模型利用这一剩余字符数估计，连同对计划中下一个词的编码特征，判断下一个词能否放进当前行。

归因图提供了算法的一种执行轨迹，展示在这条提示上哪些变量被计算、又由什么计算而来。在多样化的数据集上找到参与表示这些量的大型特征族之后，我们猜想，用低维特征流形发生几何交互的视角或许能提供更简洁的透镜。我们就以下问题找到了几何视角：

![换行行为的关键步骤可以用流形的构造与操纵来描述。](figures/linebreaks/fig-03.png)

模型如何表示不同的计数？ 词元中的字符数、当前行的字符数、整体行宽约束，以及当前行剩余的字符数，各自表示在残差流低维子空间中以高曲率嵌入的一维特征流形上。这些流形在离散特征层面有双重解释：离散特征以规范的方式铺满流形，提供近似的局部坐标。类似几何的流形会出现在多种序数概念中；在所有这些情形下，我们在嵌入几何中观察到的一种振铃（ringing）模式，相对于一个简单的物理模型而言是最优的（§表示字符计数）。[^5]

模型如何检测边界？ 为了检测逼近的行边界，模型必须比较两个量：当前字符数与行宽。我们发现了一些注意力头，其 QK 矩阵将一条计数流形旋转，使其在特定偏移下与另一条流形对齐；当两个计数之差落入目标区间时，便会产生很大的内积。多个偏移各异的注意力头协同工作，精确估计剩余字符数（§感知行边界）。

模型如何知道下一个词是否放得下？ 最终决策——是否预测换行符——需要把剩余字符数的估计与预测出的下一个词的长度结合起来。我们发现，模型将这两个计数放置在近乎正交的子空间上，从而构造出一种几何结构，使正确的换行预测在该结构中线性可分（§预测换行）。

模型如何构造这些弯曲的几何结构？ 字符计数表征流形中的曲率由许多注意力头协作产生，每个头贡献整体曲率的一部分。这种分布式算法是必要的，因为单个组件无法产生足够的输出方差来构造完整的表征（§分布式字符计数算法）。

我们通过定向干预、消融实验和"视觉错觉"来验证这些解释——所谓视觉错觉，是指劫持特定注意力机制以扰乱空间感知的字符序列（§视觉错觉）。

放宽视野，我们从这项机制案例研究中得出几条更宏观的教训：

当模型操纵流形。 要表示一个标量（例如从 $1$ 到 $N$ 的整数计数），使用 $N$ 个正交维度是低效的，而只用一维又不够有表现力[^6]。于是模型学会把这些量表示在一条内蕴维数为 1（即计数本身）的特征流形上，该流形嵌入于外蕴维度 $1 < d \ll N$ 的子空间中（例如 \cite{gorton2024curve,engels2025not,modell2025origins}），曲线在其中"起伏波动"。这种带波动的流形在容量约束（大致即维度）与保持不同标量值的可区分性（即曲率）之间实现了最优权衡。我们的工作展示了这些流形可以被操纵以执行计算的精妙方式，并说明这为何常常需要把计算分布到多个模型组件上。

特征与几何的二元性。 字典特征为发现机制提供了无监督的切入点，归因图则使任何特定预测中的重要特征浮出水面。有时，离散特征（及其相互作用）可以用连续的特征流形（及其变换）等价地描述。在能够显式参数化流形的情况下（如我们研究的各种整数计数），我们可以直接研究其几何，使某些操作（如边界检测）更加清晰。但这种方法耗费研究者的大量时间，且适用范围可能有限：研究已知的连续变量时它直截了当，但对更复杂、难以参数化的概念，正确执行就变得困难。

复杂性税。 尽管无监督发现本身就是一项胜利，字典特征仍会把模型拆解成大量细碎的部分和交互——这是加在解释之上的一种"复杂性税"。在存在流形参数化的情形下，我们可以把几何描述视为在降低这种税。在其他情形下，我们需要额外的工具来减轻解释负担，例如层级表征 \cite{costa2025flat}，或全局权重中的宏观结构 \cite{olah2023interpretability}。我们很期待看到把字典学习范式扩展到其他类型几何结构的无监督发现的方法（例如先前工作中发现的那些 \cite{hewitt2019structural,reif2019visualizing,chang2022geometry,wattenberg2024relational,park2024geometry,li2025geometry,wollschlager2025geometry,hindupur2025projecting,modell2025origins}）。

自然任务。 我们发现表征与电路之清晰利落令人相当惊叹，这可能归因于模型在该任务上表现极佳。换行对预训练语言模型而言是极其自然的行为，只要有足够的上下文，即使是微型模型也能胜任。研究对预训练语言模型而言自然的任务，而不是那些对人类研究者更具理论趣味的任务，或许能为寻找通用机制提供有前景的目标。

##### 预备知识

为了进行系统分析，我们用一个包含多样散文的文本语料库创建了合成数据集：我们（1）去除所有换行符，（2）每 $k$ 个字符在最近的词边界（$\leq k$）处重新插入换行符，其中 $k=15,20,\ldots,150$。例如，下面是葛底斯堡演说（Gettysburg Address）的开头一句，按 $k=40$ 个字符换行，换行符已显式标出。

Four score and seven years ago our⏎

fathers brought forth on this continent,⏎

a new nation, conceived in Liberty, and⏎

dedicated to the proposition that all⏎

men are created equal.

Claude 3.5 Haiku 能够适应每个 $k$ 值对应的行宽，到第三行时就能以高概率在正确位置预测换行符（见附录）。

本文正文中的所有特征均来自在 Claude 3.5 Haiku 上训练的 1000 万特征弱因果跨层转码器（Weakly Causal Crosscoder, WCC）字典 \cite{lindsey2024crosscoders}。全文的特征激活值均按各自最大值做了归一化。

#### 表示字符计数

我们定义：提示词中某个给定词元处的行字符数（line character count，或简称字符数），是指自上一个换行符以来、包括当前词元字符在内的字符总数。

一个自然的检验是：模型是否把字符数作为定量变量线性表示——也就是说，我们能否通过在残差流上做线性回归来高精度预测字符数？答案是肯定的：在第 1 层之后的残差流上拟合的线性探针，$R^2$ 达到 0.985。然而，这一成功并不意味着模型真的把字符数表示在一条直线上。

相反，我们发现字符数是一种多维表征，我们将从四个视角加以分析：

1. 稀疏跨层转码器特征。[^7]
1. 一个低维子空间。[^8]
1. 包含在该低维子空间中的连续一维流形。[^9]
1. 一组 150 个 logistic 探针（对应行字符数从 1 到 150 的取值）。[^10]

每个视角都为同一个底层对象提供了互补的视图。特征视角有助于摸清方向，子空间非常适合做因果干预，流形有助于理解表征如何被构造、又如何被操纵以检测边界，而 logistic 探针则便于分析所涉各注意力头的 OV 与 QK 矩阵。

##### 字符计数特征

我们先从特征说起。在第一层和第二层中，我们发现了似乎根据词元在行内的字符位置而激活的特征。例如，在铝提示词的归因图中，有两个特征在最后一个词"called"上处于激活状态，它们似乎分别在行字符数处于 35–55 和 45–65 之间时触发。为了找到更多这样的特征，我们按行字符数分箱计算了每个特征的平均激活值。其中有十个特征具有平滑的曲线和较大的计数间方差，如下所示：

![一族表示文本行当前字符数的特征。特征活动的调谐曲线在行字符数较大时上升。](figures/linebreaks/fig-04.png)

我们发现这些特征格外有趣，因为它们与视觉模型中的曲线检测器特征 \cite{cammarata2021curve,gorton2024missing} 以及生物大脑中的位置细胞 \cite{Moser2008PlaceCG} 颇为相似。在这三种情形中，连续变量都由一组在特定取值区间内激活的离散元素来表示。此外，我们还观察到感受野的扩张（即后续特征在越来越大的字符区间上激活），这是生物对数字感知的常见特征（例如 \cite{dehaene2003neural,piazza2004tuning}）。

在附录中，我们展示了这些特征在不同规模的字典间具有普遍性，但在行宽约束方面会出现一些特征分裂现象。

##### 模型在连续流形上表示字符数

我们观察到，字符数特征的激活曲线错位地此起彼伏，对大多数计数而言，同一时刻有两个特征处于激活状态。这一模式表明，这些特征正在重构一条弯曲的连续流形，其局部由两个最活跃特征的激活值参数化。鉴于它们的联合激活曲线呈正弦模式，我们预期重构结果位于相邻特征解码器之间的曲线上。

为了将其可视化，我们首先在合成数据集上计算每个行字符数值对应的第 2 层平均残差流。我们对这 150 个向量做主成分分析（PCA），发现前 6 个主成分捕获了 95% 的方差；我们把数据投影到这个 6 维子空间，称之为"字符计数字空间"（下图左侧为前 3 个主成分，右侧为接下来的 3 个主成分）。我们观察到数据构成一条扭转的曲线：从主成分 1–3 的视角看类似螺旋，从主成分 4–6 的视角看则呈现更复杂的扭转。

我们还仅用上面识别出的 10 个字符数特征重构每个数据点的残差流，并计算平均重构残差流。我们将得到的曲线连同特征解码器一起投影到同一子空间中。我们发现，平均行字符数向量与特征重构的结果相当接近，不过在特征向量附近存在轻微弯折，令人想起对光滑曲线的样条逼近。尽管这 10 个特征向量使曲线离散化，但在同一时刻激活的 2–3 个相邻特征之间进行插值，仍能高质量地重构这 150 个数据点。

*[字符数表示在 6 维子空间中的一条流形上（锯齿线）。这条流形可以近似地由我们识别出的特征（十字）进行局部参数化。]*

##### 验证：字符计数字空间是因果的

为了验证我们对字符计数子空间的解读，我们进行了一项粗粒度消融实验和一项细粒度干预实验。

消融实验。在消融实验中，我们从单个浅层对一个 $k$ 维子空间进行零消融——该子空间对应每个字符计数下平均激活的前 $k$ 个主成分——并与消融随机 $k$ 维子空间的基线进行对比。下面我们按换行符与非换行符分别测量损失影响。[^11]

![消融字符计数子空间仅在下一个词元是换行符时产生显著影响。](figures/linebreaks/fig-05.png)

干预实验。作为一项更为精细的干预，我们设计实验来修改 aluminum 提示词末尾感知到的字符计数（原本为 42 个字符）。具体而言，我们遍历字符计数 $c$，将数据集中所有计数为 $c$ 的词元的平均激活替换进去。即对激活 $a$ 与平均激活矩阵 $\mu$，有 $a_{\text{patched}} = a_{\text{original}} - \mu_{\text{original}} + \mu_{c}$。我们对三个相邻的浅层以及最后两个词元分别执行这一干预，既使用完整的平均向量，也使用平均向量的 6 维 PCA 空间内的向量。[^12]

![对一个秩为 6 的子空间进行干预，就足以改变模型的换行行为。](figures/linebreaks/fig-06.png)

##### 探针视角

我们还训练了有监督的逻辑回归探针来预测字符计数。[^13] 在第 1 层之后训练的探针，其均方根误差为 5，表明字符计数表示中存在一定的固有噪声——这与我们的特征具有相对较宽的感受野是一致的。对 150 个探针权重向量进行 PCA 后，我们发现 6 个主成分捕获了 82% 的方差。

当我们查看每个探针对不同行字符计数的词元的平均响应时，会看到一个引人注目的模式：除了一条对角线条带（探针与稀疏特征一样，感受野越来越宽）之外，我们还在两侧各看到两条暗淡的非对角线条带！每个探针的响应曲线在远离其最大值时并非单调递减，而是会出现回弹。这种"振铃"现象，实际上是"涟漪状"流形被嵌入低维空间的自然结果。

![行字符计数探针随行字符计数变化的响应曲线，显示出逐渐变宽的感受野以及非对角线条带的"振铃"模式。](figures/linebreaks/fig-07.png)

##### 涟漪表示是最优的

我们注意到，平均激活向量（即上文在 PCA 空间中可视化的螺旋状曲线）、线性探针向量以及特征解码器向量的余弦相似度，都呈现出与上图相似的振铃模式。[^14] 值得注意的是，不仅相邻特征之间不正交，较远的特征之间相似度为负，而更远的特征之间相似度又转为正。

![](figures/linebreaks/fig-08.png)

这种结构实际上是如下过程的自然结果：在 150 维空间中轻易就可实现的理想相似度模式，被投影到低维空间。作为此问题的一个玩具模型（toy model），假设我们希望得到一组覆盖离散化圆周的单位向量，每个向量与其相邻向量相似，而与更远的向量正交。这可以通过 150 维空间中的一组对称单位向量实现，其余弦相似度矩阵 $X$ 如下所示（左图）。将其投影到前 5 个特征向量上，就得到同一组向量的 5 维嵌入，其余弦相似度矩阵（右下图）呈现出振铃现象。我们还将这些向量在前 3 个特征向量中形成的曲线绘制了出来。我们可以把原始 150 维的圆周嵌入视为高度弯曲的，而得到的 5 维嵌入则尽可能保留了这种曲率。在 3D 投影下观察时，圆周的嵌入便呈现出涟漪状起伏。这种构造与傅里叶特征之间的关系将在附录中讨论。

![左图展示沿圆周各点对应向量的理想相似度矩阵；中图展示将这些点嵌入 5 维空间时所能达到的最优（PCA）近似；右图展示圆周在前 3 个主维度上的投影结果，呈现出涟漪状起伏。](figures/linebreaks/fig-09.png)

另一种视角是，从稀疏特征解码器的角度看，振铃可以视为一种干涉权重 \cite{olah2025toy}。在没有容量限制的情况下，模型本可以使用正交向量来表示每个特征（各有其感受野）对输入数据的定量响应。但被迫将它们压缩进低维叠加之后，相似度矩阵便同时出现了更宽的对角线条带，以及主对角线上下两侧的振铃条带。

最后，我们还构建了一个简单的物理模型，说明即使解是通过动力学方式找到的，只要大量向量被压缩进少量维度，涟漪与振铃就会出现。下面展示一次模拟的结果：100 个点被约束在 $6$ 维超球面上，受到来自两侧各 6 个最近邻点的引力（与我们的探针的 RMSE 误差相匹配），以及来自所有其他点的斥力。（为避免边界条件，我们采用圆的拓扑而非区间的拓扑。）右下图是一幅展示两条环带的热图，左下图则是这条 6 维曲线的 3 维投影。该模拟是交互式的，我们鼓励读者亲自尝试：重新初始化各点（↺）、切换环境维度、修改引力区的宽度。缩小引力区或增大嵌入维度都会提高曲率（以及振铃的程度），反之亦然。[^15] 随着曲线上点数增多、引力区宽度（相对而言）变窄，曲率会变得相当极端，在极限情况下趋近于空间填充曲线。

特别有趣的是将环境维度设为 3 的结果[^16]：得到的曲线类似于棒球的缝线（下左图，圆形），这与在 \cite{modell2025origins,engels2025not} 中观察到的三种本质上为一维的现象的拓扑相吻合——按色相排列的颜色、一年中的日期，以及 20 世纪的年份（后者还表现出扩张）。Olah 曾预言会出现类似的涟漪 \cite{olah2023manifold}，随后 Gorton 在 Inception v1 的曲线检测器特征中观察到了它们 \cite{gorton2024curve}。在余弦相似度图中观察到振铃、在低维嵌入中观察到涟漪状螺旋/螺线形状的最早案例之一，是 GPT2 中学到的词元位置嵌入 \cite{yedidia2023gpt2,yedidia2023positional}。我们在其他表示中也发现了类似结构，并将在附录的"更多感官与计数表示"中加以研究。

![左侧曲线是圆周在二维球面上的局部最优高曲率嵌入。右侧图经 Modell 等人许可转载 \cite{modell2025origins}，展示与颜色、年份和日期相关的数据或特征的 3 维 PCA 投影。](figures/linebreaks/fig-10.png)

#### 感知行边界

现在我们来研究字符计数表示如何被用于判断当前文本行是否正在接近行边界。为了检测行边界，模型需要（1）确定整体的行宽约束，（2）将当前字符计数与行宽进行比较，从而计算出剩余字符数。

##### 用 QK 扭曲

我们发现换行词元拥有自己专属的字符计数特征，这些特征根据行的宽度激活，统计相邻换行符之间的字符数量。

为了更好地理解这些表示之间的关系，我们像对"字符计数"所做的那样，为"行宽"的每个可能取值训练了 150 个探针。借助归因图，我们识别出一个激活边界检测特征的注意力头。我们直接使用两组计数表示在残差流中联合 PCA 的前 3 个分量（左图），以及该边界头约化后的 QK 空间（右图），对它们进行可视化。[^17]

*[边界头通过扭曲行宽与字符计数的表示来检测行边界。  
左：字符计数与行宽探针的联合 PCA。  
右：将它们乘以上述边界头对应的 QK 权重之后的结果。取值范围为 40（深色）到 150（浅色）。]*

我们发现这个注意力头会"扭曲"字符计数流形，使字符计数 $i$ 与行宽 $k=i+\epsilon$ 对齐。这样一来，当字符计数刚好略小于行宽时，该头就会注意换行符，从而指示边界正在接近。这一算法相当通用，使该头能够为任意行宽检测即将到来的行边界！[^18]

![行宽与字符计数探针经不同变换后的余弦相似度。（左）恒等映射；（中）边界头的 QK；（右）同层随机头的 QK。边界头使探针相互对齐，但带有较小的偏移。](figures/linebreaks/fig-11.png)

这张图表明：

- 在残差流中，当 $i=k$ 时，字符计数 $i$ 的探针与行宽 $k$ 的探针对齐程度最高，但绝对对齐程度并不高——最大余弦相似度约为 0.25。
- 在边界头的 QK 空间中，探针在非对角线区域 $i < k$ 处对齐程度最高，且绝对对齐程度接近完美——最大余弦相似度为 $\approx 1$。
- 在随机头的 QK 空间中，探针之间几乎没有结构。

作为字符计数表示中振铃现象的结果，我们在内积中也观察到了振铃（参见上文"涟漪表示是最优的"）。借助应用于注意力分数的 softmax，模型对这些非对角线干涉项具有鲁棒性。

##### 利用多个边界头

我们发现模型实际上使用了多个边界头，每个头以不同的偏移扭曲流形，从而实现一种计算剩余字符数的"立体视觉式"算法。[^19] 我们在附录中附上了更多边界头的可视化结果。

![行宽与字符计数探针经同层三个扭曲程度不同的边界头变换后的余弦相似度。绿线标示每行的 argmax，用于计算副标题中报告的平均偏移。](figures/linebreaks/fig-12.png)

为了更好地理解每个边界头的输出，我们为行内剩余字符数的每个取值训练了一组探针（即行宽 $k$ 减去字符计数 $i$，限制在 $k - i < 40$ 范围内）。对于每个边界头，我们展示了落在换行符上的注意力比例，以及每个头的输出在探针空间上的投影范数随剩余字符数的变化。

正如我们基于权重的分析所预测的那样，边界头具有各不相同但又相互重叠的响应曲线，它们"铺满"了剩余字符数的所有可能取值。

![每个边界头的响应曲线在不同距行尾距离处达到峰值。](figures/linebreaks/fig-13.png)

弄明白模型为什么需要多个边界头而不是仅仅一个，是很有价值的。如果模型只依赖边界头 0，它将无法区分剩余 5 个字符和剩余 17 个字符——两种情况会产生相似的输出。通过让每个头的输出在不同区间内变化最为显著，它们的总和就能在整个"剩余字符数"的相关取值范围内实现高分辨率。

通过在剩余字符数空间的前两个主成分（捕获了 92% 的方差）中绘制每个头的输出，我们可以更清楚地看到这一点。头 0 在 [0, 10] 和 [15, 20] 区间内表现出较大的方差，头 1 在 [10, 20] 区间内变化最大，头 2 则在 [5, 15] 区间内变化最大。虽然没有任何单个头能在整条曲线上提供高分辨率，但它们的总和产生了一个均匀分布的表示，有效地覆盖了所有取值。

![每个头的输出随剩余字符数的变化，以及它们在 PCA 基下的总和。单个头的输出几乎是一维的，而总和则是一条二维曲线。](figures/linebreaks/fig-14.png)

我们通过消融与干预实验验证了这个二维子空间的因果重要性。具体来说，我们进行了与之前相同的实验：消融该子空间并按词元测量其对损失的影响（左图），以及通过替换平均激活向量来精确调节 aluminum 提示词中最后一个词元上的剩余字符数估计（右图）。

![剩余字符数子空间可被因果性地干预。（左）消融该子空间仅在下一个词元是换行符时产生显著影响。（右）我们对剩余字符数空间进行精细干预：减去真实的剩余字符数平均激活，再加入修补后的剩余字符数激活，从而调节对换行符的预测。注意补全词 " aluminum." 需要十个字符才能放下。](figures/linebreaks/fig-15.png)

##### 额外维度的作用

现在我们可以回答两个既不同又相关的问题：（1）为什么这些计数表示是多维的；（2）为什么计算这些多维表示需要多个注意力头。

几何计算——多维表示使模型能够利用线性变换旋转位置编码，这是用一维表示无法做到的。例如，为了检测即将到来的行边界，模型可以旋转位置流形使其与行宽对齐，然后通过点积来判断何时只剩下几个字符。对于 1D 编码，线性运算退化为缩放和平移，因此将位置与行宽进行比较只会把两个值相乘，产生一个单调递增、没有自然阈值的结果。2D 以上的更高维度允许流形通过额外的曲率封装更多信息。

分辨率——对于字符计数，模型必须在大范围的字符位置上区分相邻计数，因为这决定了下一个词是否放得下。在一维表示中，位置将沿一条射线排列，每个位置之间相隔某个常数 $\delta$。为了在噪声之上可靠地区分相邻位置，我们需要 $||v_{42} - v_{41}|| = \delta$ 超过某个阈值。但要表示 150 多个位置，这就造成了一个两难选择：要么使用巨大的动态范围（$||v_{150}|| \gg ||v_1||$），这对 transformer 的计算来说是有问题的；要么牺牲相邻位置之间的分辨率。（归一化模块只会加剧这一问题：虽然只要范数足够大，点就可以在射线上彼此相距很远，但该射线在单位超球面上的投影至多只有 $\pi$ 的角距离。）将曲线嵌入更高维度解决了这个问题：各位置保持相近的范数，同时在外围空间中彼此良好分离，在不发生范数爆炸的情况下实现了精细的分辨率（参见上文"涟漪表示是最优的"）。对于剩余字符数的计数，动态范围更小，因此模型得以将表示嵌入一个更小的子空间。

要达到高分辨率所需的曲率，需要多个注意力头协同构造计数流形的弯曲几何。单个注意力头的输出是其输入的线性组合（按注意力加权，并经 OV 电路变换），因而从根本上受限于输入中已有的曲率。在计数表示没有 MLP 贡献的情况下，如果输出流形需要表现出显著的曲率，多个注意力头就必须协调配合——每个头贡献整体几何结构的一部分。在"分布式字符计数算法"一节中，我们还会看到分布式头计算的另一个例子。

##### 发现的故事

我们最初是如何发现这种边界检测机制的？当我们第一次计算归因图时，看到了若干条从"上一个换行"特征和嵌入指向"预测换行"特征的边。QK 归因显示，最关键的键特征是"上一行长度为 40–60 个字符"特征，最关键的查询特征是"当前字符计数为 35–50"特征。在任意时刻，往往有多个计数特征以不同强度同时激活，这提示我们：这些特征可能正在离散化一个流形。

![](figures/linebreaks/fig-16.png)

边界头会促使一族边界检测特征根据当前行与全局行宽的接近程度而激活。也就是说，它们感知的是逼近的行边界，或者说行计数的反向索引。考察这三组特征族群，让我们找到了它们稀疏参数化的计数流形；而考察相关的注意力头，则让我们发现了边界头。

最后，我们注意到，这些边界感知表示与神经科学中一个被广泛研究的现象相呼应：边界细胞 \cite{solstad2008representation}——它们在距环境边界（如墙壁）特定距离处激活。人工特征与生物细胞都以族群形式出现，具有各不相同的感受野与偏移。

#### 预测换行

换行任务的最后一步，是把对行边界的估计与对下一个词的预测结合起来，判断下一个词能否排进当前行，还是应该断行。

在铝提示词的归因图中，我们恰好看到这种路径的汇合。整张图中最具影响力的特征[^20]是一个深层特征，它在"下一个词将导致当前行超过整体行宽"的语境中激活。就我们的提示词而言，该特征上调换行的概率、下调"aluminum"的概率。这个换行预测特征的两大输入，是"say aluminum"特征和由前述边界头激活的"边界检测"特征。

![](figures/linebreaks/fig-17.png)

边界检测器无论下一词元多长都会激活，而换行预测特征只在下一词元将超过当前行长度时才会激活（正如铝提示词中的情形），因此上调换行的预测。[^21] 我们还看到换行抑制特征，它们只在下一词元刚好勉强能排进当前行时激活，因此下调换行的预测。换行预测器与抑制器都出自更大的特征族群，我们在附录中展示。

![三个特征的平均激活，按真实下一词元字符长度与行内剩余字符数（行宽 − 字符计数）分组。](figures/linebreaks/fig-18.png)

##### 联合几何使计算变得简单

模型判断下一词元能否排进当前行的能力，其底层几何是什么？换句话说，上面的换行预测特征是如何由边界检测器与下一词特征构造出来的？

为了研究这一点，我们计算模型末端（约 90% 深度）所有词元上的平均激活，覆盖剩余字符数 $i$ 与下一词元长度 $j$ 的全部取值。[^22] 对均值向量的组合做 PCA 后，我们看到两个计数分布在正交子空间中，曲率适中。注意，这里低维的几何或许就足够了，因为计数的动态范围要小得多。

*[下一词元字符长度与剩余字符计数流形的低维投影，对应 1（深色）至 15（浅色）个字符。  
（左）二者并集的 PCA。（右）所有两两组合的 PCA。  
正交的表示使正确的换行决策线性可分。]*

现在考虑每个可能的剩余字符向量 $i$ 与下一词元长度向量 $j$ 的两两之和。[^23] 由于这两个计数正交排列，是否断行的决策 $i-j \geq 0$ 便对应一个简单的分离超平面。换句话说，断行预测被底层几何变得轻而易举！

在真实数据上使用这些平均嵌入的 PCA 分离超平面时，对于"下一词元是否应为换行"的真值，我们达到了 0.91 的 AUC。这既反映了三维分类器的误差，也反映了 Haiku 对下一词元估计的误差。

如果最可能的下一词的长度被线性表示，那么这一方案就能让模型在该词长于行内剩余长度时预测换行。我们可以设想一种更通用的机制：模型把所有超过行限的词的概率质量整体转移到换行上。Claude 3.5 Haiku 似乎并未利用这样的机制：当我们比较行末词元的预测分布与去掉换行后的相同提示词上的分布时，发现两者差异相当大。

#### 分布式字符计数算法

在描述了各种字符计数表示的用途之后，剩下的最后一个大问题是：它们是如何计算出来的？

我们将展示 Haiku 如何利用跨越多个层的众多注意力头，协同计算出越来越精确的字符计数估计。这是我们研究过的最复杂的机制，尽管它与边界检测机制有许多相似之处。

为了直观理解对计数重要的头的行为，我们把它们的输出投影到行字符计数探针的 PCA 空间中。[^24] 层 0 的头（左）各自沿射线写入——在前 3 个主成分中可视化时，它们看起来就是射线——而正是这些头的和生成了弯曲的流形。层 1 的头（右）则输出曲线，这些曲线组合成越来越复杂的流形。它们似乎负责锐化层 0 的表示，从而锐化计数的估计。我们发现，5 个关键的层 0 头对字符计数预测 [^25] 的 $R^2$ 为 0.93，而使用前两层共 11 个头时为 0.97。

*[层 0（左）与层 1（右）的平均注意力输出在字符计数探针 PCA 基下的对比，对应 1（深色）至 150（浅色）个字符。每一层中，各头的输出铺满整个空间。  
层 0 中，每个头的输出几乎是 1 维的；层 1 中的头则展现出更多曲率（这些曲率来自层 0！）。]*

##### 嵌入的几何

为了理解字符计数是如何计算的，我们从最源头开始：嵌入矩阵。

与之前一样，我们可以训练探针，或计算嵌入中每个不同词元长度的平均权重。我们可视化字符长度 1–14 的词元字符计数探针，并可视化它们的前几个主成分。使用捕获了 70% 方差的前 3 个主成分，我们看到嵌入中的字符计数呈环形排列（PC1 对 PC2），并带有振荡分量（PC3）。这一模式与"涟漪表示是最优的"中观察到的模式一致。

![按词元字符长度平均的 $W_E$ 中嵌入向量的 PCA。](figures/linebreaks/fig-19.png)

与所有计数流形一样，我们也发现了将这一空间离散化为"短词、中词、长词"等相互重叠概念的特征。

##### 注意力头输出求和产生计数

为了理解计数机制，我们将从求和后的注意力输出反向追溯到嵌入。具体来说，我们：

- 忽略 MLP——注意力头输出对字符计数表示的影响是 MLP 的 4 倍以上，因此我们把关注点限定在注意力上；
- 聚焦前两层——即使在层 0 之后，计数探针就已具有不错的准确度，并且存在粗粒度的位置特征。因此，我们关注注意力如何把嵌入变换为计数，以及层 1 如何进一步精化这一表示。

![5 个重要的层 0 头在单个提示上的按词元求和输出。（左）求和注意力输出与字符计数探针的内积；（右）该内积的 argmax 与真实行计数的对比。上下文位置从第一个换行开始，换行以短划线标记。](figures/linebreaks/fig-20.png)

我们可以把上面的和分解为层 0 中每个头各自输出的贡献。[^26] 在这个视角下，我们看到每个头都在执行一种相对低秩的计算，类似于分类。

![4 个重要的层 0 头在单个提示上的各自输出，投影到字符计数探针上。](figures/linebreaks/fig-21.png)

单个头是如何实现这一行为的？我们可以通过分析单个头的 QK 电路（它在哪里分配注意力）与 OV 电路（从嵌入到输出的线性变换），来拆解其行为 \cite{nelhage2021mathematical}。

QK 电路。每个头 $h$ 都把上一个换行当作"注意力汇"（attention sink）：在换行之后的若干词元（$s_h$ 个）内，头只关注换行。超过 $s_h$ 个词元后，头开始在它的感受野上摊开注意力，最多覆盖 $r_h$ 个词元。

![对上一个换行的平均注意力，随行内词元索引变化。与边界头类似，这些计数头以不同的位置偏移实现特化。](figures/linebreaks/fig-22.png)

OV 电路。OV 电路与 QK 电路协同，基于行内词元数乘以平均词元长度（$\mu_c \approx 4$）产生一个启发式估计，并附有一个额外的长度修正项。当关注换行时，每个头上调"平均词元长度乘以该头的汇大小"：$s_h\times\mu_c$ 个字符。如果没有关注换行，那么在头看来，当前词元至少已在行内 $s_h+r_h$ 个词元处，因此应上调 $(s_h+r_h) \times \mu_c$ 个字符的输出。最后，OV 电路还会根据感受野内的词元长度是高于还是低于平均水平，施加一项额外修正。

下面，我们给出 L0H1 的详细讲解。

![计数头 L0H1 的 QK 与 OV 电路。右上：单个提示 64 个词元的头输出（截断至第一个换行）投影到字符计数探针上。右下：注意力模式（规范排序的转置）。左上：平均嵌入向量经 OV 矩阵投影到字符计数探针上。左下：整体计算的总结。](figures/linebreaks/fig-23.png)

关于每个头更详细的分析，参见"头部特化的机制"。层 1 的注意力头执行类似的操作，但额外利用了字符计数的初步估计（见"层 1 头的 OV 电路"）。

##### 计算行宽

为了计算行宽，模型似乎使用了类似的分布式计数算法来统计相邻换行之间的字符数。不过，本文没有解决的一个微妙之处是行宽究竟如何聚合。模型可能是取文档中所有行长度的最大值来计算全局行宽，也可能使用最近几行长度的指数加权移动平均。我们确实注意到，行宽使用了一组部分不相交的头，这很可能是因为当当前词元本身也是换行时，"把上一个换行当作汇来关注"的机制需要加以修改。

#### 视错觉

人类容易受到"视错觉"的影响：语境线索能以看似出人意料的方式调节感知。著名的例子包括缪勒-莱尔错觉（Müller-Lyer illusion）——在线段两端放置箭头可以改变线段被感知的长度 \cite{howe2005muller}；庞佐错觉（Ponzo illusion）与桑德错觉（Sander illusion），它们同样调节被感知的线长 \cite{yildiz2022review}；以及其他错觉 \cite{schwartz2007space}。

![线长感知被调制的经典视错觉。](figures/linebreaks/fig-24.png)

我们能利用对字符计数机制的理解，为语言模型构造一种"视错觉"吗？

作为起点，我们选取了字符计数中重要的注意力头，调查它们在更广泛的数据分布上还扮演哪些其他角色。我们找到了这样的实例：通常从换行出发关注上一个换行的头，转而从换行出发关注两字符字符串 @@。这个字符串在 git diff 中用作分隔符——在这种情况下，你可能希望从换行以外的位置开始计行：

⏎@@-14,30 +31,24 @@ export interface ClaudeCodeIAppTheme {⏎

但如果这个序列出现在 git diff 语境之外，会发生什么？例如，在保持行长不变的情况下向铝提示词中插入 @@ 呢？

![](figures/linebreaks/fig-25.png)

我们发现它确实调节了预测的下一词元，扰乱了换行预测！正如所料，相关的头被"分心"了：在原始提示词中，头从换行关注到换行；而在改动后的提示词中，头还会关注 @@。

![插入 @@ “分散”了一个通常从 \n 回看上一个 \n 的注意力头的注意力。（左）原始注意力模式（截断）。（右）插入 @@ 后的注意力模式（截断）。现在它也会回看 @@。](figures/linebreaks/fig-26.png)

这一结果有多特异：在提示词中无意义插入的任意一对字母，都会彻底扰乱换行预测吗？我们分析了在相同的两个位置插入 180 种不同两字符序列的影响，其中一半是重复字符。我们发现，虽然大多数插入的序列会适度影响预测换行的概率，但换行通常仍是头号预测。由相同字符或不同字符构成的序列之间，也没有明显差异。不过，少数序列显著扰乱了换行预测，其中大多数似乎与某种代码或分隔符有关：``  >>  }}  ;|  ||  `,  @@.

我们进一步分析了重要注意力头的"分心"程度与换行预测所受影响之间的关系。事实上，我们发现许多对换行概率有强烈调制的序列——尤其是与代码相关的字符对——其注意力模式也发生了显著调制。

![插入大多数字符对只会适度影响预测换行的概率；其中一部分字符对（大多与代码或分隔符有关）会显著干扰换行预测。对换行预测的影响（原本为 0.79）与插入词元对字符计数注意力头的"分心"程度相关。](figures/linebreaks/fig-27.png)

虽然在铝提示词中任务是隐含的，但这种错觉也能推广到比较任务被明确化的场景。这些直接的比较或许更接近庞佐（Ponzo）、桑德（Sander）与缪勒-莱尔（Müller-Lyer）错觉——在这些错觉中，感知与比较更为直接。

![](figures/linebreaks/fig-28.png)

这些效应不受选项排列顺序的影响。此外，如果 @@ 之后文本的长度超过备选项的长度，备选项就会被判定为更短。

我们并非声称人类视觉感知错觉与这种对行字符数估计的篡改之间存在直接类比，但两者的相似之处确实引人深思。在这两种情形中，我们都能看到更广泛的现象：上下文线索，以及关于这些线索的习得先验，会调制对实体对象属性的估计。在人类的情形中，三维透视等先验会影响对物体大小的感知，颜色恒常性则会影响对亮度的估计（如棋盘阴影错觉）。在这里，我们结果的一种可能解释是：习得先验的误用——包括诸如 git diff 中 @@ 这类线索所起的作用——同样会调制对行长度等属性的估计。

#### 相关工作

目标。 本工作处于 LLM"生物学"（对模型内部正在发生什么做经验性观察；例如 \cite{lindsey2025biology,rogers2020primer}）与神经网络底层逆向工程（试图完整刻画某个算法或机制；例如 \cite{olah2020zoom,wang2022interpretability,nanda2023progress,li2025language}）的交汇处。在方法论上，我们的工作大量使用归因图 \cite{ameisen2025circuit,ge2024automatically,dunefsky2024transcoders}，并在跨层转码器 \cite{lindsey2024crosscoders} 之上构建 QK 归因 \cite{kamath2025tracing}。

换行。 Michaud 等人 \cite{michaud2023the} 将等宽文本中的换行识别为 Pythia 系列中最小模型（7000 万参数）行为中排名前 400 的"量子"之一。

位置。 以往关于位置机制的可解释性研究大多聚焦于词元位置（例如 \cite{yedidia2023gpt2,yedidia2023positional,voita2023neurons,chughtai2024understanding,gurnee2024universal}）。这些工作表明，存在具有周期结构、编码绝对词元位置的 MLP 神经元 \cite{voita2023neurons,gurnee2024universal}、SAE 特征 \cite{chughtai2024understanding} 与习得的位置嵌入 \cite{yedidia2023gpt2}。我们的工作则展示了模型如何也可能构造非词元式的位置方案——这种方案对许多下游预测任务而言更为自然。

也有其他研究者（甚至可以追溯到 LSTM 时代）研究过语言模型中控制输出响应长度的机制 \cite{shi2016neural,moon2025length}，并对计数算法的空间做过更理论化的分析 \cite{suzgun2019lstm,chang2024language}。

几何与特征流形。 除位置之外，还有大量工作致力于理解数字的几何表示，尤其是在玩具模型（toy model）中（例如 \cite{nanda2023progress,zhong2023clock,morwani2023feature}）以及 LLM 算术的语境下（例如 \cite{stolfo2023mechanistic,zhou2024pre,nikankin2024arithmetic,kantamneni2025trig,hu2025understanding}）。这些工作共同表明，真实 LLM 与玩具 transformer 都会习得周期表示 \cite{zhou2024pre,kantamneni2025trig,hu2025understanding}，数字呈螺旋排列，以支持某些基于矩阵乘法的加法算法 \cite{nanda2023progress,kantamneni2025trig}，而且在某些设定下这些表示可证明是最优的 \cite{morwani2023feature}。在我们的情境中，我们同样观察到螺旋表示 \cite{kantamneni2025trig}、数值膨胀 \cite{alquboj2025number}，以及各组件之间分布式协同、共同实现正确计算的算法 \cite{hanna2023does,hu2025understanding}。

在更自然的情境中，人们也发现了具有清晰几何结构的多维特征 \cite{gould2023successor,engels2025not,modell2025origins}，例如某些序数关系的表示与计算（比如一年中的月份）。在视觉模型中，曲线检测神经元 \cite{cammarata2020curve} 与曲线检测特征 \cite{gorton2024missing} 得到了尤其充分的研究，它们与我们观察到的字符计数特征族中的离散化方式十分相似。许多其他主题也已获得对其底层几何的可解释性分析，例如语法关系 \cite{hewitt2019structural,reif2019visualizing}、多语言表示 \cite{chang2022geometry}、真值 \cite{marks2023geometry}、绑定 \cite{feng2023language}、拒绝 \cite{wollschlager2025geometry}、特征 \cite{hindupur2025projecting,li2025geometry} 与层级结构 \cite{park2024geometry}，不过仍需要更多概念层面的研究 \cite{wattenberg2024relational}。

也许最相关的是 Modell 等人 \cite{modell2025origins} 的近期工作：他们给出了特征流形更形式化的定义，并提出余弦相似度编码了特征的内在几何。在检验他们的理论时，他们观察到高度结构化且可解释的数据流形，带有波纹与膨胀，与我们的计数流形相似。这些观察提出了一个方法论挑战——如何最好地捕捉具有不同结构的数据（参见例如 \cite{hindupur2025projecting,michaud2025understanding,huang2025decomposing}）——同时也引出一个令人振奋的假说：许多天然连续的变量（例如 \cite{heinzerling2024monotonic,gurnee2024language}）存在于更有组织的流形之中。

生物学类比。 我们观察到的几何与算法模式，与生物神经系统的感知有着耐人寻味的相似之处。我们的字符计数特征类似于一维轨道上的位置细胞 \cite{Moser2008PlaceCG}，而我们的边界检测特征类似于边界细胞 \cite{solstad2008representation}。这些特征表现出膨胀——越来越大的字符计数在越来越大的范围内激活——与生物大脑中数字表征的膨胀如出一辙 \cite{dehaene2003neural,piazza2004tuning}。此外，特征在低维流形上的组织，是生物认知中一种常见模式的具体实例（例如 \cite{perich2025neural}）。虽然这些类比并不完美，但我们相信，神经科学与可解释性之间加强合作仍能产生丰硕的概念重叠 \cite{vilas2024position,he2024multilevel,leshinskaya2025cognitively}。

#### 讨论

在本文中，我们研究了一个大模型执行自然行为所涉及的各个步骤。换行任务在训练中经常遇到，它要求模型以字符数为单位表示并计算若干与位置有关的标量——这些量在其输入或输出中并不显式存在[^27]——然后把这些数值与复杂语义电路（预测下一个恰当单词）的输出整合起来，预测下一个词元。我们找到了与计算中每个重要步骤相对应的稀疏特征；对于那些涉及标量的步骤，我们得以找到一种几何描述，它显著简化了对模型所用算法的解读。现在我们来反思从这一过程中得到的认识：

自然行为与感觉处理。 深入的机制案例研究宜选择模型一贯表现良好的行为，因为这类行为更可能具有更清晰的机制。这意味着应当优先选择在预训练中自然的任务，而不是那些对人类研究者显得自然的任务；理想情况下，这些任务还应易于监督。与生物神经科学一样，感知任务往往既自然，又易于为可解释性研究提供监督（例如，以编程方式修改输入很容易）。尽管我们有时把语言模型的早期层描述为负责"解词元化"输入 \cite{elhage2022solu,gurnee2023finding,ferrando2024information,lad2024remarkable}，但把它视为感知或许更具启发性。模型的起点真正负责的是"看见"输入，早期电路中有很大一部分都在为感知文本服务——就像视觉模型的早期层实现低级感知一样 \cite{olah2020zoom,lepori2024beyond}。

几何的效用。 我们研究的许多表示与计算都有优雅的几何解释。例如，计数流形是容量与分辨率之间最优权衡的结果，与空间填充曲线和傅里叶特征有着深刻的联系。边界头的扭转尤其优美：在发现这样一个头之后，我们得以正确预测出，还需要额外的头来为输出提供曲率。分布式字符计数算法更为复杂，但我们仍能通过研究这些流形上的线性作用来澄清我们的理解。对于其他计算，比如最终的断行决策，线性可分显然是其中的一部分，但必定还存在某种我们尚未能看见的额外复杂性，用以处理多词元输出。对于更偏语义的操作，我们则完全依赖特征视角。当然，完整描述任何行为都极其复杂，还有一长串我们未研究的潜在细微之处：模型如何应对计数中的不确定性；面对多行先前的文本，它估算行宽的机制；它如何适应行宽可变的文档；它如何处理多个长度不同或由多个词元构成的合理候选输出；以及各种特殊情况（例如 LaTeX 的 \footnote{} 或 Markdown 链接）。对于有兴趣的读者，我们使用新的 Neuronpedia 交互界面，分享了等宽文本换行提示词的转码器归因图，适用于 [Gemma 2 2B](https://www.neuronpedia.org/gemma-2-2b/graph?slug=fourscoreandseve-1757368139332&pruningThreshold=0.8&densityThreshold=0.99&pinnedIds=14_19999_37&clerps=%5B%5B%2214_200290090_37%22%2C%22nearing+end+of+the+line%22%5D%5D) 与 [Qwen 3 4B](https://www.neuronpedia.org/qwen3-4b/graph?slug=fourscoreandseve-1757451285996&pruningThreshold=0.8&densityThreshold=0.99&clerps=%5B%5B%2230_117634760_39%22%2C%22nearing+end+of+line%22%5D%5D&pinnedIds=30_15307_39)。

无监督发现。 若非这些无监督的稀疏特征，我们很可能无法获得如此清晰的理解。事实上，项目开始时，我们曾试图仅靠探针探测与激活修补来摸索理解，但结果并不理想。具体而言，我们不知道自己要找什么（例如，我们不知道要区分行宽与字符数）、到哪里去找（例如，我们没想到行宽只在换行符上表示）、以及怎么去找（我们起初训练的是一维线性回归探针）。然而，在识别出一些相关特征之后、在投入大量精力系统刻画它们的激活模式之前，我们对它们所表示的内容也感到困惑。我们看到几十个隐约与换行和换行文本有关的特征，但仅靠翻阅激活示例，它们的差异并不明显。直到我们在合成数据集上检验这些特征之后，它们在归因图中的角色以及底层计算才变得清晰。我们推测，更好的自动标签 \cite{bills2023language,paulo2024automatically,gur2025enhancing} 配合智能体工作流 \cite{shaham2024multimodal,bricken2025automating} 会加速这类工作，尤其是在较难验证的领域。

特征-流形二元性。 离散特征视角与几何的特征流形视角，是观察同一个底层对象的双重透镜。例如，在本工作中，模型对字符数的表示可以完全由我们识别出的特征的活动来描述（忽略重构误差），其中边界头的作用由虚拟权重描述——这些虚拟权重通过注意力头矩阵展开特征之间的交互。同一个字符数表示也可以描述为一维特征流形——残差流中一条以字符数变量为参数的曲线——其中边界头的线性作用被描述为对流形的连续"扭转"。一般而言，模型习得的几何结构很可能同时容许全局参数化与局部离散近似这两种描述。

复杂性税。 尽管存在这种二元性，两种视角给出的描述在简洁性上仍有差异。离散特征把模型碎裂成许多碎片，产生对计算的复杂理解。这似乎是一条普遍的经验。离散特征与归因图似乎能提供对模型计算的真实描述，而借助字典学习可以自动化地找到这种描述。能得到对计算的任何真实而可理解的描述，都是非常来之不易的胜利！然而，如果我们止步于此，不去理解其中存在的额外结构，就要付出复杂性税——以一种不必要的复杂方式去理解事物。在换行问题中，构造流形偿还了这笔税，但我们也可以设想其他减轻解释负担的途径。

对方法论的呼吁。 凭借对特征的理解，我们得以直接搜索相关的几何结构。这更像一个存在性证明，而非通用配方；我们需要能够自动浮现更简单结构的方法，来偿还复杂性税。在我们的情境中，这意味着研究特征流形，我们希望能看到检测它们的无监督方法。在其他情境中，我们还需要其他工具来减轻解释负担，例如在全局权重 \cite{ameisen2025circuit} 中寻找层级表示 \cite{costa2025flat} 或宏观结构 \cite{olah2023interpretability}。

对生物学的呼吁。 模型一定还在执行其他优雅的计算。我们可以从模型擅长的某个具体任务入手，从多个视角研究它，发展方法论来回答剩余的问题，并坚持不懈地尝试简化我们的解释。由于这类研究扎根于某一行为的具体实例，它提供了快速的反馈回路，能揭示现有方法的弱点并催生新方法，还能磨砺我们理解神经网络的概念语言。我们期待看到更多采用这种方法的深度案例研究。

## 脚注

[^1]: 所有特征都有一个幅值维度；因此离散特征是一条一维射线，而一维特征流形是该流形所有缩放形式的集合，收缩至原点。参见[什么是线性表示？什么是多维特征？](https://transformer-circuits.pub/2024/july-update/index.html#linear-representations)

[^2]: Michaud 等人通过聚类梯度寻找模型技能的"量子" \cite{michaud2023the}。他们的图 1 显示，对 Pythia 家族中最小模型（7000 万参数）而言，预测等宽文本中的换行构成了前 400 大聚类之一。

[^3]: 换行约束是隐式的。每个换行都给出一个下界（上一个单词确实放得下）和一个上界（下一个单词放不下）。我们并不确定模型在多大程度上针对这些约束进行最优推断，而是专注于它如何近似地利用前面每一行的长度来决定是否断开下一行。此外，词元化与标点的处理还有许多边界情况。模型甚至可能试图推断源文档是否使用了非等宽字体，进而用像素数而非字符数作为预测信号！

[^4]: 我们最初其实尝试过不看归因图，直接使用激活修补与探针探测，作为检验特征实用性的一种方法论测试，但进展不大。事后看来，我们当时训练的探针所针对的量，与模型清晰表征的量并不相同——例如，当前词元位置与行宽的融合。

[^5]: 从流形视角看，振铃（ringing）对应于特征叠加视角中的干涉。

[^6]: 正交维度同样无法稳健地抵抗估计噪声。

[^7]: 每个特征都有一个编码器和一个解码器，其中编码器在残差流上充当线性 + (Jump)ReLU 探针。十个特征 $f_1,\ldots,f_{10}$ 与行字符数相关联。给定残差流向量 $x$，模型对字符数的估计由这 10 个特征各自的激活值集合 $\{f_i(x)\}$ 概括。

[^8]: 模型对字符数的估计由 $x$ 在该子空间上的投影 $\pi(x)$ 概括。若两个数据点的投影在该子空间中彼此接近，则它们的字符数相近。

[^9]: 模型对字符数的估计由流形上离 $x$ 在该子空间中的投影最近的点概括，而模型对该估计的置信度则由 $\pi(x)$ 的大小概括。

[^10]: 模型对字符数的估计由探针激活值的 softmax 给出的概率分布概括，即 softmax$(Px)$。

[^11]: 请注意，一般而言，不应假设由特征（或 PCA）张成的子空间专属于这些特征，因为它可能与其他许多特征处于叠加状态。然而，在本例中字符数子空间是密集激活的（因此不太容易参与叠加），所以这一实验设计更具合理性。

[^12]: 归因图在最后一个词元（"called"）和倒数第二个词元（"also"）上都有若干位置特征与边。为保持一致性，我们将"also"的计数表示改为最终词元计数之前 6 个字符处。

[^13]: 作为一个 150 类多分类问题

[^14]: 我们按信号处理的意义使用"[振铃（ringing）](https://en.wikipedia.org/wiki/Ringing_(signal))"一词，指对尖锐峰值产生的瞬态振荡，例如吉布斯现象（Gibbs Phenomenon）中的情形。

[^15]: 模拟有时会陷入局部极小值。先将吸引区宽度增大、再将其减小，通常可以解决这个问题。

[^16]: 与更高维度不同，三维中的优化会存在不良局部极小值，因为球面上的一般曲线会自相交。为避免这一问题，可以先将吸引区宽度增大直到得到大圆（great circle），再将其减小；也可以在四维中完成优化，然后再选定三维结果。

[^17]: 具体来说，我们将行宽探针乘以 $W_K$、将字符数探针乘以 $W_Q$，然后在二者联合嵌入的三维 PCA 基中绘制这些点。

[^18]: 该算法还能推广到任意类型的分隔符（例如双换行或竖线），因为 QK 电路可以独立处理位置偏移，而 OV 电路负责复制分隔符类型。

[^19]: 多层中还存在多组边界头，通常以约 3 个为一组、具有相近的相对偏移（因此并非真正的"立体声"）。

[^20]: 影响指对 logit 节点的影响，定义见 Ameisen et al. \cite{ameisen2025circuit}

[^21]: 这些特征有时也会在零宽修饰词元上激活（例如，指示下一个词元首字母应大写的词元）；此时该修饰词元须紧邻被修饰词元，且被修饰词元要足够长、足以超出行宽限制（例如 "Aluminum" 而非 "aluminum"）。

[^22]: 我们使用真实的下一个非换行词元作为标签。这是一种近似，因为它假设模型能完美预测下一个词元。

[^23]: 这一求和是有依据的，因为两组向量都是边际化的数据均值，合在一起便得到数据的均值，我们将其中心化为 0。

[^24]: 我们展示的是许多提示词上的平均输出。

[^25]: 预测即头输出投影到字符数探针上的 argmax。

[^26]: 为便于视觉呈现，我们省略了一个前一词元头。

[^27]: 词元本身并不附带字符数标注，页面上也没有显示行宽的竖线。

## 参考文献

- [olah2023manifold]: Olah, Christopher, Batson, Josh, “Feature Manifold Toy Model”, 2023
- [olah2024multidimensional]: Olah, Christopher, “What is a Linear Representation? What is a Multidimensional Feature?”, 2024
- [gorton2024curve]: Gorton, Liv, “Curve Detector Manifolds in InceptionV1”, 2024
- [engels2025not]: Joshua Engels, Eric J Michaud, Isaac Liao, Wes Gurnee, Max Tegmark, “Not All Language Model Features Are One-Dimensionally Linear”, The Thirteenth International Conference on Learning Representations, 2025
- [michaud2023the]: Eric J Michaud, Ziming Liu, Uzay Girit, Max Tegmark, “The Quantization Model of Neural Scaling”, Thirty-seventh Conference on Neural Information Processing Systems, 2023
- [ameisen2025circuit]: Ameisen, Emmanuel, Lindsey, Jack, Pearce, Adam, Gurnee, Wes, Turner, Nicholas L., Chen, Brian, Citro, Craig, Abrahams, David, Carter, Shan, Hosmer, Basil, Marcus, Jonathan, Sklar, Michael, Templeton, Adly, Bricken, Trenton, McDougall, Callum, Cunningham, Hoagy, Henighan, Thomas, Jermyn, Adam, Jones, Andy, Persic, Andrew, Qi, Zhenyi, Ben Thompson, T., Zimmerman, Sam, Rivoire, Kelley, Conerly, Thomas, Olah, Chris, Batson, Joshua, “Circuit Tracing: Revealing Computational Graphs in Language Models”, Transformer Circuits, 2025
- [modell2025origins]: Modell, Alexander, Rubin-Delanchy, Patrick, Whiteley, Nick, “The Origins of Representation Manifolds in Large Language Models”, arXiv preprint arXiv:2505.18235
- [costa2025flat]: Costa, Val{\'e}rie, Fel, Thomas, Lubana, Ekdeep Singh, Tolooshams, Bahareh, Ba, Demba, “From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit”, arXiv preprint arXiv:2506.03093
- [olah2023interpretability]: Olah, Chris, “Interpretability Dreams”, Transformer Circuits Thread, 2023
- [hewitt2019structural]: Hewitt, John, Manning, Christopher D, “A structural probe for finding syntax in word representations”, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019
- [reif2019visualizing]: Coenen, Andy, Reif, Emily, Yuan, Ann, Kim, Been, Pearce, Adam, Viégas, Fernanda, Wattenberg, Martin, “Visualizing and measuring the geometry of BERT”, Advances in Neural Information Processing Systems, 2019
- [chang2022geometry]: Chang, Tyler A, Tu, Zhuowen, Bergen, Benjamin K, “The geometry of multilingual language model representations”, arXiv preprint arXiv:2205.10964
- [wattenberg2024relational]: Wattenberg, Martin, Vi{\'e}gas, Fernanda B, “Relational composition in neural networks: A survey and call to action”, arXiv preprint arXiv:2407.14662
- [park2024geometry]: Park, Kiho, Choe, Yo Joong, Jiang, Yibo, Veitch, Victor, “The geometry of categorical and hierarchical concepts in large language models”, arXiv preprint arXiv:2406.01506
- [li2025geometry]: Li, Yuxiao, Michaud, Eric J, Baek, David D, Engels, Joshua, Sun, Xiaoqing, Tegmark, Max, “The geometry of concepts: Sparse autoencoder feature structure”, Entropy, 2025
- [wollschlager2025geometry]: Wollschl{\"a}ger, Tom, Elstner, Jannes, Geisler, Simon, Cohen-Addad, Vincent, G{\"u}nnemann, Stephan, Gasteiger, Johannes, “The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence”, arXiv preprint arXiv:2502.17420, 2025
- [hindupur2025projecting]: Hindupur, Sai Sumedh R, Lubana, Ekdeep Singh, Fel, Thomas, Ba, Demba, “Projecting assumptions: The duality between sparse autoencoders and concept geometry”, arXiv preprint arXiv:2503.01822
- [lindsey2024crosscoders]: Lindsey, Jack, Templeton, Adly, Marcus, Jonathan, Conerly, Thomas, Batson, Joshua, Olah, Christopher, “Sparse Crosscoders for Cross-Layer Features and Model Diffing”, 2024
- [cammarata2021curve]: Cammarata, Nick, Goh, Gabriel, Carter, Shan, Voss, Chelsea, Schubert, Ludwig, Olah, Chris, “Curve Circuits”, Distill, 2021
- [gorton2024missing]: Gorton, Liv, “The Missing Curve Detectors of InceptionV1: Applying Sparse Autoencoders to InceptionV1 Early Vision”, arXiv preprint arXiv:2406.03662, 2024
- [Moser2008PlaceCG]: Edvard I. Moser, Emilio Kropff, May-Britt Moser, “Place cells, grid cells, and the brain's spatial representation system.”, Annual review of neuroscience, 2008
- [dehaene2003neural]: Dehaene, Stanislas, “The neural basis of the Weber--Fechner law: a logarithmic mental number line”, Trends in cognitive sciences, 2003
- [piazza2004tuning]: Piazza, Manuela, Izard, V{\'e}ronique, Pinel, Philippe, Le Bihan, Denis, Dehaene, Stanislas, “Tuning curves for approximate numerosity in the human intraparietal sulcus”, Neuron, 2004
- [olah2025toy]: Olah, Chris, Turner, Nicholas L., Conerly, Tom, “A Toy Model of Interference Weights”, 2025
- [yedidia2023gpt2]: Yedidia, Adam, “GPT-2's positional embedding matrix is a helix”, 2023
- [yedidia2023positional]: Yedidia, Adam, “The positional embedding matrix and previous-token heads: how do they actually work?”, Alignment Forum, 2023
- [solstad2008representation]: Solstad, Trygve, Boccara, Charlotte N, Kropff, Emilio, Moser, May-Britt, Moser, Edvard I, “Representation of geometric borders in the entorhinal cortex”, Science, 2008
- [nelhage2021mathematical]: Elhage, Nelson, Nanda, Neel, Olsson, Catherine, Henighan, Tom, Joseph, Nicholas, Mann, Ben, Askell, Amanda, Bai, Yuntao, Chen, Anna, Conerly, Tom, DasSarma, Nova, Drain, Dawn, Ganguli, Deep, Hatfield-Dodds, Zac, Hernandez, Danny, Jones, Andy, Kernion, Jackson, Lovitt, Liane, Ndousse, Kamal, Amodei, Dario, Brown, Tom, Clark, Jack, Kaplan, Jared, McCandlish, Sam, Olah, Chris, “A Mathematical Framework for Transformer Circuits”, Transformer Circuits Thread, 2021
- [howe2005muller]: Howe, Catherine Q, Purves, Dale, “The M\"uller-Lyer illusion explained by the statistics of image--source relationships”, Proceedings of the National Academy of Sciences, 2005
- [yildiz2022review]: Yildiz, Gizem Y, Sperandio, Irene, Kettle, Christine, Chouinard, Philippe A, “A review on various explanations of Ponzo-like illusions”, Psychonomic Bulletin \& Review, 2022
- [schwartz2007space]: Schwartz, Odelia, Hsu, Anne, Dayan, Peter, “Space and time in visual context”, Nature Reviews Neuroscience, 2007
- [lindsey2025biology]: Lindsey, Jack, Gurnee, Wes, Ameisen, Emmanuel, Chen, Brian, Pearce, Adam, Turner, Nicholas L., Citro, Craig, Abrahams, David, Carter, Shan, Hosmer, Basil, Marcus, Jonathan, Sklar, Michael, Templeton, Adly, Bricken, Trenton, McDougall, Callum, Cunningham, Hoagy, Henighan, Thomas, Jermyn, Adam, Jones, Andy, Persic, Andrew, Qi, Zhenyi, Thompson, T. Ben, Zimmerman, Sam, Rivoire, Kelley, Conerly, Thomas, Olah, Chris, Batson, Joshua, “On the Biology of a Large Language Model”, Transformer Circuits Thread, 2025
- [rogers2020primer]: Rogers, Anna, Kovaleva, Olga, Rumshisky, Anna, “A primer in bertology: What we know about how bert works”, Transactions of the Association for Computational Linguistics, 2020
- [olah2020zoom]: Olah, Chris, Cammarata, Nick, Schubert, Ludwig, Goh, Gabriel, Petrov, Michael, Carter, Shan, “Zoom In: An Introduction to Circuits”, Distill, 2020
- [wang2022interpretability]: Wang, Kevin, Variengien, Alexandre, Conmy, Arthur, Shlegeris, Buck, Steinhardt, Jacob, “Interpretability in the wild: a circuit for indirect object identification in gpt-2 small”, arXiv preprint arXiv:2211.00593, 2022
- [nanda2023progress]: Nanda, Neel, Chan, Lawrence, Lieberum, Tom, Smith, Jess, Steinhardt, Jacob, “Progress measures for grokking via mechanistic interpretability”, arXiv preprint arXiv:2301.05217, 2023
- [li2025language]: Li, Belinda Z, Guo, Zifan Carl, Andreas, Jacob, “(How) Do Language Models Track State?”, arXiv preprint arXiv:2503.02854
- [ge2024automatically]: Ge, Xuyang, Zhu, Fukang, Shu, Wentao, Wang, Junxuan, He, Zhengfu, Qiu, Xipeng, “Automatically identifying local and global circuits with linear computation graphs”, arXiv preprint arXiv:2405.13868, 2024
- [dunefsky2024transcoders]: Dunefsky, Jacob, Chlenski, Philippe, Nanda, Neel, “Transcoders find interpretable LLM feature circuits”, Advances in Neural Information Processing Systems, 2025
- [kamath2025tracing]: Kamath, Harish, Ameisen, Emmanuel, Kauvar, Isaac, Luger, Rodrigo, Gurnee, Wes, Pearce, Adam, Zimmerman, Sam, Batson, Joshua, Conerly, Thomas, Olah, Chris, Lindsey, Jack, “Tracing Attention Computation Through Feature Interactions”, Transformer Circuits Thread, 2025
- [voita2023neurons]: Voita, Elena, Ferrando, Javier, Nalmpantis, Christoforos, “Neurons in large language models: Dead, n-gram, positional”, arXiv preprint arXiv:2309.04827
- [chughtai2024understanding]: Chughtai, Bilal, Lau, Yeu-Tong, “Understanding positional features in layer 0 SAEs”, 2024
- [gurnee2024universal]: Gurnee, Wes, Horsley, Theo, Guo, Zifan Carl, Kheirkhah, Tara Rezaei, Sun, Qinyi, Hathaway, Will, Nanda, Neel, Bertsimas, Dimitris, “Universal neurons in gpt2 language models”, arXiv preprint arXiv:2401.12181, 2024
- [shi2016neural]: Shi, Xing, Knight, Kevin, Yuret, Deniz, “Why neural translations are the right length”, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing
- [moon2025length]: Moon, Sangjun, Choi, Dasom, Kwon, Jingun, Kamigaito, Hidetaka, Okumura, Manabu, “Length Representations in Large Language Models”, arXiv preprint arXiv:2507.20398
- [suzgun2019lstm]: Suzgun, Mirac, Gehrmann, Sebastian, Belinkov, Yonatan, Shieber, Stuart M, “LSTM networks can perform dynamic counting”, arXiv preprint arXiv:1906.03648
- [chang2024language]: Chang, Yingshan, Bisk, Yonatan, “Language models need inductive biases to count inductively”, arXiv preprint arXiv:2405.20131
- [zhong2023clock]: Zhong, Ziqian, Liu, Ziming, Tegmark, Max, Andreas, Jacob, “The clock and the pizza: Two stories in mechanistic explanation of neural networks”, Advances in neural information processing systems, 2023
- [morwani2023feature]: Morwani, Depen, Edelman, Benjamin L, Oncescu, Costin-Andrei, Zhao, Rosie, Kakade, Sham, “Feature emergence via margin maximization: case studies in algebraic tasks”, arXiv preprint arXiv:2311.07568
- [stolfo2023mechanistic]: Stolfo, Alessandro, Belinkov, Yonatan, Sachan, Mrinmaya, “A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis”, arXiv preprint arXiv:2305.15054, 2023
- [zhou2024pre]: Zhou, Tianyi, Fu, Deqing, Sharan, Vatsal, Jia, Robin, “Pre-trained large language models use Fourier features to compute addition”, arXiv preprint arXiv:2406.03445, 2024
- [nikankin2024arithmetic]: Yaniv Nikankin, Anja Reusch, Aaron Mueller, Yonatan Belinkov, “Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics”, 2024
- [kantamneni2025trig]: Subhash Kantamneni, Max Tegmark, “Language Models Use Trigonometry to Do Addition”, 2025
- [hu2025understanding]: Hu, Xinyan, Yin, Kayo, Jordan, Michael I, Steinhardt, Jacob, Chen, Lijie, “Understanding In-context Learning of Addition via Activation Subspaces”, arXiv preprint arXiv:2505.05145
- [alquboj2025number]: AlquBoj, HV, AlQuabeh, Hilal, Bojkovic, Velibor, Hiraoka, Tatsuya, El-Shangiti, Ahmed Oumar, Nwadike, Munachiso, Inui, Kentaro, “Number Representations in LLMs: A Computational Parallel to Human Perception”, arXiv preprint arXiv:2502.16147
- [hanna2023does]: Hanna, Michael, Liu, Ollie, Variengien, Alexandre, “How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model”, Advances in Neural Information Processing Systems, 2023
- [gould2023successor]: Rhys Gould, Euan Ong, George Ogden, Arthur Conmy, “Successor Heads: Recurring, Interpretable Attention Heads In The Wild”, 2023
- [cammarata2020curve]: Cammarata, Nick, Goh, Gabriel, Carter, Shan, Schubert, Ludwig, Petrov, Michael, Olah, Chris, “Curve Detectors”, Distill, 2020
- [marks2023geometry]: Marks, Samuel, Tegmark, Max, “The geometry of truth: Emergent linear structure in large language model representations of true/false datasets”, arXiv preprint arXiv:2310.06824
- [feng2023language]: Feng, Jiahai, Steinhardt, Jacob, “How do language models bind entities in context?”, arXiv preprint arXiv:2310.17191, 2023
- [michaud2025understanding]: Eric J. Michaud, Liv Gorton, Tom McGrath, “Understanding sparse autoencoder scaling in the presence of feature manifolds”, 2025
- [huang2025decomposing]: Huang, Xinting, Hahn, Michael, “Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning”, arXiv preprint arXiv:2508.01916
- [heinzerling2024monotonic]: Heinzerling, Benjamin, Inui, Kentaro, “Monotonic representation of numeric properties in language models”, arXiv preprint arXiv:2403.10381
- [gurnee2024language]: Wes Gurnee, Max Tegmark, “Language Models Represent Space and Time”, 2024
- [perich2025neural]: Perich, Matthew G, Narain, Devika, Gallego, Juan A, “A neural manifold view of the brain”, Nature Neuroscience, 2025
- [vilas2024position]: Vilas, Martina G, Adolfi, Federico, Poeppel, David, Roig, Gemma, “Position: An inner interpretability framework for AI inspired by lessons from cognitive neuroscience”, arXiv preprint arXiv:2406.01352
- [he2024multilevel]: He, Zhonghao, Achterberg, Jascha, Collins, Katie, Nejad, Kevin, Akarca, Danyal, Yang, Yinzhu, Gurnee, Wes, Sucholutsky, Ilia, Tang, Yuhan, Ianov, Rebeca, others, “Multilevel interpretability of artificial neural networks: leveraging framework and methods from neuroscience”, arXiv preprint arXiv:2408.12664
- [leshinskaya2025cognitively]: Leshinskaya, Anna, Webb, Taylor, Pavlick, Ellie, Feng, Jiahai, Opielka, Gustaw, Stevenson, Claire, Blank, Idan A, “Cognitively Inspired Interpretability in Large Neural Networks”, Proceedings of the Annual Meeting of the Cognitive Science Society
- [elhage2022solu]: Elhage, Nelson, Hume, Tristan, Olsson, Catherine, Nanda, Neel, Henighan, Tom, Johnston, Scott, ElShowk, Sheer, Joseph, Nicholas, DasSarma, Nova, Mann, Ben, Hernandez, Danny, Askell, Amanda, Ndousse, Kamal, Jones, And, Drain, Dawn, Chen, Anna, Bai, Yuntao, Ganguli, Deep, Lovitt, Liane, Hatfield-Dodds, Zac, Kernion, Jackson, Conerly, Tom, Kravec, Shauna, Fort, Stanislav, Kadavath, Saurav, Jacobson, Josh, Tran-Johnson, Eli, Kaplan, Jared, Clark, Jack, Brown, Tom, McCandlish, Sam, Amodei, Dario, Olah, Christopher, “Softmax Linear Units”, Transformer Circuits Thread, 2022
- [gurnee2023finding]: Gurnee, Wes, Nanda, Neel, Pauly, Matthew, Harvey, Katherine, Troitskii, Dmitrii, Bertsimas, Dimitris, “Finding Neurons in a Haystack: Case Studies with Sparse Probing”, arXiv preprint arXiv:2305.01610, 2023
- [ferrando2024information]: Ferrando, Javier, Voita, Elena, “Information flow routes: Automatically interpreting language models at scale”, arXiv preprint arXiv:2403.00824
- [lad2024remarkable]: Lad, Vedang, Lee, Jin Hwa, Gurnee, Wes, Tegmark, Max, “The remarkable robustness of llms: Stages of inference?”, arXiv preprint arXiv:2406.19384
- [lepori2024beyond]: Lepori, Michael, Tartaglini, Alexa, Vong, Wai Keen, Serre, Thomas, Lake, Brenden M, Pavlick, Ellie, “Beyond the doors of perception: Vision transformers represent relations between objects”, Advances in Neural Information Processing Systems
- [bills2023language]: Bills, Steven, Cammarata, Nick, Mossing, Dan, Tillman, Henk, Gao, Leo, Goh, Gabriel, Sutskever, Ilya, Leike, Jan, Wu, Jeff, Saunders, William, “Language models can explain neurons in language models”, 2023
- [paulo2024automatically]: Paulo, Gon{\c{c}}alo, Mallen, Alex, Juang, Caden, Belrose, Nora, “Automatically interpreting millions of features in large language models”, arXiv preprint arXiv:2410.13928
- [gur2025enhancing]: Gur-Arieh, Yoav, Mayan, Roy, Agassy, Chen, Geiger, Atticus, Geva, Mor, “Enhancing automated interpretability with output-centric feature descriptions”, arXiv preprint arXiv:2501.08319
- [shaham2024multimodal]: Shaham, Tamar Rott, Schwettmann, Sarah, Wang, Franklin, Rajaram, Achyuta, Hernandez, Evan, Andreas, Jacob, Torralba, Antonio, “A multimodal automated interpretability agent”, Forty-first International Conference on Machine Learning
- [bricken2025automating]: Bricken, Trenton, Wang, Rowan, Bowman, Sam, Ong, Euan, Treutlein, Johannes, Wu, Jeff, Hubinger, Evan, Marks, Samuel, “Building and evaluating alignment auditing agents”, 2025
