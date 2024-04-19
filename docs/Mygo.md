# MyGO: Discrete Modality Information as Fine-Grained Tokens for Multi-modal Knowledge Graph Completion

### 摘要：

多模态知识图（MMKG）存储了包含丰富多模态描述信息的结构化世界知识。为了克服它们固有的不完整性，多模态知识图补全（MMKGC）旨在利用结构信息和实体的多模态信息，从给定的 MMKG 中发现未观测到的知识。现有的 MMKGC 方法通常使用预训练模型提取多模态特征，并采用融合模块将多模态特征与三元组预测结合起来。然而，这种方法常常导致对多模态数据的粗略处理，忽略了精细的语义细节及其交互。为了解决这一不足，我们引入了一个新颖的框架 MyGO，用于处理、融合和增强 MMKG 中的精细模态信息。MyGO 将多模态原始数据标记化为精细的离散标记，并通过跨模态实体 encoder 学习实体表征。为了进一步增强多模态表征，MyGO 加入了精细的对比学习，以突出实体表征的特异性。在标准 MMKGC 基准测试上的实验显示，我们的方法超过了 20 种最新模型，凸显了其卓越的性能。代码和数据可在 https://github.com/zjukg/MyGO 获取。

### 1.Introduction

![](https://arxiv.org/html/2404.09468v1/x1.png)

> (a) 精细的多模态语义单元及其对应关系。

![](https://arxiv.org/html/2404.09468v1/x2.png)

> (b) 现有 MMKGC 方法和 MyGO 的直觉理解。

图 1. MMKG 的多模态信息中存在精细的语义交互。MyGO 试图将原始模态数据标记化为精细的 token，并利用多模态 token 序列学习精细的实体表征。

多模态知识图（MMKGs）（Chen 等，2024b）将多样化和复杂的世界知识以结构化三元组（头实体、关系、尾实体）的形式封装，同时结合图像和文本等多模态数据以提供额外的实体上下文。这些广泛的三元组及其多模态内容共同形成了一个庞大的多模态语义网络，为推荐系统（Sun 等，2020）、多模态理解（Zhu 等，2021）和大型语言模型（Dong 等，2024；Chen 等，2023b）等多个领域提供了重要的基础设施。MMKGs 为这些系统提供了一个可靠的事实知识来源。

MMKGs 经常面临不完整性的挑战，因为在其创建过程中仍有大量有效知识未被发现。这一现象突显了**多模态知识图补全**（MMKGC）（Chen 等，2024b）的重要性，其目标是从给定的 MMKGs 中自动识别新知识。与传统的知识图补全（KGC）（Bordes 等，2013；Sun 等，2019）主要侧重于基于现有 KGs 模拟三元组结构不同，MMKGC 需要管理丰富实体描述的额外多模态信息。**因此，MMKGC 的本质是将来自三元组的结构信息与与实体相关的丰富多模态特征和谐地整合**。这种协同作用对于在 embedding 空间内进行有根据的知识推理至关重要，其中实体的丰富多模态信息作为补充信息，为三元组预测提供了坚实有效的多模态特征。

现有的 MMKGC 方法（Sergieh 等，2018；Cao 等，2022；Li 等，2023；Lee 等，2023）倾向于将模态信息表示为来自预训练模型（Devlin 等，2019）的单一 embedding，并利用融合和预测模块来衡量三元组的可信度。然而，这一范式相当简单化，常常无法捕捉模态数据中存在的复杂细节。通常在这一范式中，由预训练模型提取的模态信息在后续训练中会被冻结。此外，在处理多个模态实例（如一个实体的多张图片）时，这些方法会采用普通操作如平均，从而丢失可能重要的细节。考虑到原始模态数据包含了用于展示关键实体特征的详细语义单元，通常生成每种模态的静态 embedding 的做法可能导致宝贵的粒度信息的损失，进而限制 MMKGC 模型的性能。例如，我们在图 1（a）中展示了一个简单的案例，展示了 T-Rex 实体的图像和文本中的这些精细语义单元，即图像段和文本短语。这些精细的语义特征不仅描述了一个实体，还体现了复杂的跨模态关系。我们提倡一个更精细的框架，允许 MMKGC 模型通过详细的交互捕获数据中 embedding 的微妙、共享信息。这种方法有望显著增强实体表征。

为了解决精细信息处理和利用问题，我们提出了一个新颖的框架 MyGO，以实现 MMKGC 模型中的精细多模态信息处理、交互和增强。图 1（b）清晰地对比了现有 MMKGC 方法和我们的 MyGO。**MyGO 首先采用模态标记化（MT）模块，使用现有的预训练标记器**（Devlin 等，2019；Peng 等，2022）将 MMKGs 中的实体模态信息标记化为精细的离散 token 序列，**然后通过分层三元组建模（HTM）架构学习 MMKGC 任务。HTM 包括一个跨模态实体 encoder、一个上下文三元组 encoder 和一个关系 decoder**，用于编码精细的实体表征并衡量三元组的可信度。为了进一步增强和细化实体表征，**我们提出了一个精细的对比损失（FGCL），以生成多样化的对比样本并提升模型性能**。我们在公共 MMKG 基准（Liu 等，2019；Xu 等，2022）上进行了全面的实验。与 20 个最近的基线方法相比，比较结果显示 MyGO 的性能更优。我们还进一步深入探讨了 MyGO 设计的细节以便更好地理解。我们的贡献有三方面：

- 我们强调 MMKGC 的精细多模态学习，并提出了尖端框架 MyGO。MyGO 将模态数据标记化为精细的多模态 token，并开创了一个新颖的 MMKGC 架构，以分层建模跨模态实体表征。
- 我们提出了一个精细对比学习模块，以增强跨模态实体表征学习过程。这个模块通过采用新策略生成高质量的对比样本，创新了更详细和有效的自我监督对比学习。
- 我们在公共基准上进行了全面的实验，并在 MMKGC 上实现了针对 20 个经典基线方法的新的最佳性能。我们通过广泛的实验进行了进一步的探索。

![](https://arxiv.org/html/2404.09468v1/x3.png)

图 2. 我们的 MyGO 框架概览。在 MyGO 中，我们主要有三个新设计部分来处理、交互和增强 MMKGs 中的精细多模态语义信息。这三个部分分别是模态标记化、分层三元组建模和精细对比学习。

### 2. 相关工作

#### 2.1 多模态知识图补全

**多模态知识图（MMKGs）（Chen 等人，2024b，2023a）是富含诸如图像、文本描述、音频和视频等多模态信息的知识图**（Wang 等人，2023）。由于知识图的不完整性，知识图补全（KGC）（Bordes 等人，2013; Ji 等人，2015; Yang 等人，2015; Trouillon 等人，2016; Sun 等人，2019）成为一个热门的研究话题，**目标是通过学习三元组结构来自动发现未观测的知识三元组。** 多模态知识图补全（MMKGC）旨在利用实体的额外多模态信息协同预测给定 MMKGs 中缺失的三元组。

现有的 MMKGC 方法主要从三个角度进行新的改进：（1）多模态融合与交互，（2）集成决策，（3）负采样。第一类方法（Xie 等人，2017; Sergieh 等人，2018; Wang 等人，2021; Cao 等人，2022; Xu 等人，2023; Chen 等人，2024a; Zhang 等人，2024a）设计复杂机制以在表示空间中实现多模态融合与交互。例如，OTKGE 提出了一种基于最优传输的多模态融合策略，以找到多模态融合的最佳权重。第二类方法（Zhao 等人，2022; Li 等人，2023）通常为每种模态学习一个判别模型，并将它们集成以作出联合决策。IMF（Li 等人，2023）提出了一种交互式多模态融合方法，学习四种不同模态信息的 MMKGC 模型以实现联合决策。第三类方法（Xu 等人，2022; Zhang 等人，2023a, b, 2024b）旨在通过实体的多模态信息增强负采样过程（Bordes 等人，2013），生成高质量的负样本。总体而言，这些 MMKGC 方法通常利用多模态信息通过从预训练模型（Simonyan 和 Zisserman，2015; Devlin 等人，2019）提取特征表示。然而，特征处理往往忽略了每种模态中的精细语义信息。我们将通过将模态信息标记化为精细 token 来解决这一问题。

#### 2.2 多模态信息标记化

标记化是 NLP 领域广泛使用的一种技术，用于将输入文本处理成 token 序列，并学习字符串和子词的精细文本表示。由于文本模态本身的特性，标记化非常有效，已广泛用于语言模型（LM）。例如，BPE（Gage，1994）、WordPiece（Wu 等人，2016）和 ULM（Kudo，2018）是最著名的标记化方法。对于其他模态的信息，由于这些模态没有清晰的分割点，标记化相对困难，与文本不同。向量量化（VQ）（van den Oord 等人，2017; Esser 等人，2021）是一个重要的技术，提出将大规模数据映射到固定长度的离散码本，其中每个码本中的代码都是代表某些特定特征的向量。因此，非文本模态信息可以首先被处理成补丁序列，然后每个补丁映射到一个离散的代码，这可以被视为多模态 token，并进一步在许多任务中得到利用（Peng 等人，2022; Ryoo 等人，2021）。VQ 的优点是在保留广泛的精细模态特征的同时压缩多模态数据。例如，BEIT-v2（Peng 等人，2022）将每幅图像处理成 196 个大小为 16x16 的补丁，并将每个补丁映射到一个离散代码。在我们的工作中，我们也将采用 VQ 和标记化来处理 MMKGs 中的多模态信息，获得实体的精细多模态表征。

### 3. 任务定义

一个融合了视觉和文本模态的多模态知识图（MMKG）可以表示为 $\mathcal{G}=(\mathcal{E}, \mathcal{R}, \mathcal{T}, \mathcal{V}, \mathcal{D})$，其中 $\mathcal{E}$ 和 $\mathcal{R}$ 分别是实体集和关系集。$\mathcal{J}=\{(h, r, t) \mid h, t \in \mathcal{E}, r \in \mathcal{R}\}$ 是三元组集合，表示实体 $h$ 通过关系 $r$ 与实体 $t$ 相关联。此外，$\mathcal{V}, \mathcal{D}$ 分别对应每个实体 $e$ 的图像集合和文本描述。

知识图补全（KGC）的主要目标是**学习一个评分函数 $\delta(h, r, t): \mathcal{E} \times \mathcal{R} \times \mathcal{E} \rightarrow \mathbb{R}$，该函数通过一个标量分数来衡量三元组 $(h, r, t)$ 的可能性**。在 KGC 模型中，实体和关系对应于 embedding，并且三元组的分数是基于这些 embedding 定义的，偏好正三元组的高分和负三元组的低分。换句话说，**通过训练中的正负对比（Bordes 等人，2013）最大化训练集中正三元组的可能性**。扩展到 MMKGs，多模态知识图补全（MMKGC）将进一步考虑每个实体 $e$ 的多模态信息 $\mathcal{V}(e), \mathcal{D}(e)$ 来增强它们的 embedding。现代方法通常为每种模态创建 embedding，并整合这些 embedding 来计算三元组分数，采用各种多模态融合技术（Cao 等人，2022; Wang 等人，2021; Li 等人，2023）以提升性能。

在推理阶段，MMKGC 模型将预测给定查询 $(h, r, ?)$ 或 $(?, r, t)$ 的缺失实体。以尾部预测 $(h, r, ?)$ 为例，MMKGC 模型将把 $\mathcal{E}$ 中的每个实体 $e$ 视为候选实体，并计算其对应的分数 $(h, r, e)$。此外，模型的评估将通过金标准答案 $(h, r, t)$ 对所有候选者的排名来进行，这意味着将采用基于排名的指标（Bordes 等人，2013; Sun 等人，2019）来评估性能。对于头部预测也是类似的，总体性能通常考虑测试三元组上的头部和尾部预测。

### 4. 方法论

在本节中，我们将详细介绍我们提出的框架 MrGO（简称 MrGO），该框架利用模态信息作为精细化的单元（ModalitY information as fine-Grained tO kens）。我们采用了主流的多模态知识图补全（MMKGC）设置（Xu 等人，2022），该设置包括图像和文本模态（Chen 等人，2024b）。MrGO 主要包括三个模块：模态标记化模块、分层三元组建模模块和精细对比学习模块，分别旨在处理、融合和增强 MMKGs 中的精细信息。图 2 直观地展示了 MrGO 的设计。

#### 4.1 模态标记化

为了捕获精细的多模态信息，我们提出了一个模态标记化（MT）模块，**将实体的原始多模态数据处理成精细的离散语义 token**，作为学习精细实体表征的语义单元。我们分别使用**针对图像和文本模态的标记器**，分别表示为 $Q_{i m g}$ 和 $Q_{t x t}$，为实体 $e$ 生成视觉 token $v_{e, i}$ 和文本 token $w_{e, i}$：

$$
\begin{aligned}
& U_{i m g}(e)=\left\{v_{e, 1}, v_{e, 2}, \cdots, v_{e, m_e}\right\}=Q_{i m g}(\mathcal{V}(e)) \\
& U_{t x t}(e)=\left\{w_{e, 1}, w_{e, 2}, \cdots, w_{e, n_e}\right\}=Q_{t x t}(\mathcal{D}(e))
\end{aligned}
$$

其中 $m_{\varepsilon}, n_{\varepsilon}$ 是每种模态的 token 数量，我们用 $U(e)$ 表示实体 $e$ 的集合 token 集。文本 token 来自语言模型的词汇表（Devlin 等人，2019），而视觉 token 来自预训练视觉标记器的码本（Esser 等人，2021; Peng 等人，2022）。值得注意的是，$\mathcal{V}(e)$ 可能包含多个图像，我们处理每个图像以累积 $U_{i m g}(e)$ 中的 token。

在标记化过程中，由于某些子词可能在句子中多次出现，以及类似的语义元素可能在实体图像中重复出现，常常会遇到重复的 token。**因此，我们计算每个 token 的出现频率，为每种模态保留预定数量的最常见 token**。此外，我们还去除了文本描述中的停用词（Wilbur 和 Sirotkin，1992），因为它们对实体语义的贡献最小。经过这种精细化处理后，我们为 MMKG 中的每个实体保留 $m$ 个视觉 token 和 $n$ 个文本 token。对于 token 不足或模态数据缺失的实体，我们添加一个特殊的填充 token 来填补空白。经过 MT 和精细化处理后，我们可以获得每个实体 $e$ 的处理过的 token 集 $\mathcal{U}_{i m g}^{\prime}$ 和 $\mathcal{U}_{t x t}^{\prime}$，这些 token 集包含了从原始多模态数据中提取的关键特征的精细 token。随后，我们为 $u_{i m g}^{\prime}$ 和 $u_{t x t}^{\prime}$ 中的每个 token 分配一个单独的 embedding。这种方法考虑到不同实体可能共享 token，个性化的 tokenembedding 允许跨不同实体更精细地表示类似特征，通过详细的多模态语义单元丰富实体的描述。

与现有的 MMKGC 方法不同，MT 技术将模态信息转换成更精细的离散 token。当面对单一模态中的多重信息（例如，一个实体的多张图片）时，传统的 MMKGC 方法会先进行聚合处理（例如，平均化处理（张等人，2024b））。然而，MT 保留了一系列代表各种原始数据源中最常见特征的 token，这对于增加模态信息来说可以更稳定和可扩展。我们将在实验中演示这一点。

#### 4.2 分层三元组建模

在完成 MT 处理后，我们在本节进一步设计了一个分层三元组建模（HTM）模块。HTM 利用分层 transformer 架构来捕获多模态实体表征，并以逐步方式建模三元组的可能性，它由三个部分组成：跨模态实体 encoder、上下文三元组 encoder 和关系 decoder。

##### 4.2.1 跨模态实体 encoder

跨模态实体 encoder（CMEE）旨在通过利用其精细的多模态 token 来捕获实体的多模态表征。与现有方法（曹等人，2022 年；张等人，2024b 年；李等人，2023 年）不同，这些方法用单一 embedding 表示每种模态然后设计融合策略来合并它们，MrGO 执行不同模态的精细标记化并获得每种模态的 token 序列。因此，我们设计了一种更精细的特征交互方法，允许所有不同模态消息之间的全面交互。在 MrGO 中，我们使用 Transformer（Vaswani 等人，2017 年）层作为 CMEE。我们首先将多模态 token 线性化为序列 $X$：

$$
X(e)=\left([\mathrm{ENT}], s_e, v_{e, 1}, \cdots, v_{e, m}, w_{e, 1}, \cdots, w_{e, n}\right)
$$

其中 $[\mathrm{ENT}]$ 是一个特殊 token，$s_e$ 是一个可学习的 embedding，代表实体的结构信息。[ENT] 类似于 BERT 中的 [CLS] token（Devlin 等人，2019），用来捕获用于下游预测的序列特征。$s_e$ 是一个可学习的 embedding，代表从现有三元组结构中学到的结构信息，这将在训练期间进行优化。此外，对于来自 $u_{i m g}^{\prime}$ 和 $u_{t x t}^{\prime}$ 的多模态 token，我们冻结它们从标记器中得到的初始表征，并定义线性投影层 $\mathcal{P}_{\text{img}}, \mathcal{P}_{\text{txt}}$，将它们投影到相同的表征空间：

$$
\hat{v}_{e, i}=\mathcal{P}_{i m g}\left(v_{e, i}\right)+b_{i m g} \quad \hat{w}_{e, j}=\mathcal{P}_{t x t}\left(v_{e, j}\right)+b_{t x t}
$$

其中 $b_{\text{img}}, b_{t x t}$ 是定义的模态偏置，用于增强来自不同模态的信息标记。我们的目标不是调整基础 token 特征，而是通过训练投影层来改善它们的整合，以实现更好的泛化。这样，输入 CMEE 的最终序列变为：

$$
X_{\text{input}}(e)=\left

([ENT], s_e, \hat{v}_{e, 1}, \cdots, \hat{v}_{e, m}, \hat{w}_{e, 1}, \cdots, \hat{w}_{e, n}\right)
$$

##### 4.2.2 上下文三元组 encoder

为了在关系上下文中实现充分的模态交互，我们应用另一个 transformer 层作为上下文三元组 encoder（CTE），用于编码给定查询的上下文 embedding。以头查询（$h, r, ?$）（尾部预测）为例，我们可以获得上下文 embedding $\tilde{\mathbf{h}}$：

$$
\widetilde{\mathbf{h}}=\operatorname{Transformer}([\mathrm{CXT}], \mathbf{h}, \mathbf{r})
$$

其中 $[\mathrm{CXT}]$ 是输入序列中的特殊令牌，用于捕获实体的上下文 embedding，$\mathbf{h}$ 是 $h$ 从 CMEE 得到的输出表示，$\mathbf{r}$ 是每个 $r \in \mathcal{R}$ 的关系 embedding。然后通过关系 decoder 处理查询 $(h, r, ?)$ 的上下文 embedding，以进行实体预测。

##### 4.2.3 关系 decoder

此外，我们采用得分函数 $\mathcal{S}(h, r, t)$ 来通过产生标量分数来衡量三元组的可能性，这个函数作为查询预测的关系 decoder。在 MYGO 中，我们采用 Tucker（Balazevic 等人，2019）作为我们的得分函数，表示为：

$$
\mathcal{s}(h, r, t)=\mathcal{W} \times_1 \tilde{\mathbf{h}} \times_2 \mathbf{r} \times_3 \mathbf{t}
$$

其中 $x_i$ 表示沿着第 $\mathrm{i}$ 模的张量积，$\mathcal{W}$ 是在训练期间学习的核心张量。我们使用交叉熵损失来训练我们的模型。我们将 $t$ 视为针对整个实体集 $\mathcal{E}$ 的金标准标签，这同样适用于头部预测。因此，训练目标是交叉熵损失：

$$
\mathscr{L}_{\text {head }}=-\sum_{(h, r, t) \in \mathcal{T}} \log \frac{\exp (\mathcal{S}(h, r, t))}{\sum_{t^{\prime} \in \varepsilon^{\prime}} \exp \left(\mathcal{S}\left(h, r, t^{\prime}\right)\right)}
$$

请注意，我们使用 $h$ 的上下文 embedding $\widetilde{\mathbf{e}}_h$ 和 $t$ 的多模态 embedding $\mathbf{e}_t$ 来计算得分，这可以加快计算速度。否则，我们需要提取所有候选实体在不同关系下的上下文 embedding，这需要 $O(|\mathcal{E}| \times|\mathcal{R}|)$ 级的上下文 transformer 前向传递，将大大增加模型的计算量。此外，MYGO 同时考虑头部和尾部预测，当给出尾查询 $(?, r, t)$ 时，目标 $\mathscr{L}_{\text {tail }}$ 也类似。

（10）

$$
\mathscr{L}_{\text {tail }}=-\sum_{(h, r, t) \in \mathcal{T}} \log \frac{\exp (\mathcal{S}(h, r, t))}{\sum_{h^{\prime} \in \varepsilon^{\prime}} \exp \left(\mathcal{S}\left(h^{\prime}, r, t\right)\right)}
$$

整个多模态知识图补全（MMKGC）任务目标可以表示为：

$$
\mathscr{L}_{\text {kgc }}=\mathscr{L}_{\text {head }}+\mathscr{L}_{\text {tail }}
$$

#### 4.3 细粒度对比学习

基于上述设计，我们已能够训练和测试 MMKGC 模型。为了进一步增强细粒度和健壮的多模态实体表征，我们在 MrGO 中引入了一个细粒度对比学习（FGCL）模块，通过对实体表征的多尺度对比学习来实现这一目标。

如前所述，跨模态实体 encoder（CMEE）旨在基于多模态令牌序列捕捉实体表征。受到 SimCSE（高等，2021）的启发，我们通过对比学习增强这些实体表征。具体来说，给定一个实体 $e$，我们可以通过两次前向传播从 CMEE 获得两个表征 $\mathbf{e}, \mathbf{e}_{sec}$。这两个 embedding 之间的变化，由 transformer encoder 中的 dropout 层引起，允许对多模态令牌特征进行轻微的去激活，有效地起到了简单数据增强的作用。通过批次内的对比学习，MYGO 被训练以从令牌序列中提取真正重要的信息，从而增强每个实体表征的独特性。为了加深这一过程的细粒度，我们进一步从 transformer 输出中提取三个附加表征，可以从各自的视角代表实体特征。我们可以定义输入序列 $x_{\text {input }}(e)$ 中多模态令牌的输出表征为：

$$
x_{\text {output }}(e)=\left([\mathrm{ENT}]^{\prime}, s_e^{\prime}, \hat{v}_{e, 1}^{\prime}, \cdots, \hat{v}_{e, m}^{\prime}, \hat{w}_{e, 1}^{\prime}, \cdots, \hat{w}_{e, n}^{\prime}\right)
$$

然后我们引入三个 embedding $\mathbf{s}(e), \mathbf{v}(e), \mathbf{w}(e)$ 来代表一个实体 $e$ 的全局、视觉和文本信息。$\mathbf{s}(e)$ 是从 $X_{\text {output }}(e)$ 中所有输出表征的平均值得到的。同样，$\mathbf{v}(e)$ 和 $\mathbf{w}(e)$ 是相应视觉和文本令牌的平均值。它们可以表示为：

（13）

$$
\mathbf{s}(e)=\operatorname{Mean}\left(X_{\text {output }}(e)\right)
$$

$$
\mathbf{v}(e)=\frac{1}{m} \sum_{i=1}^m \hat{v}_{e, i}^{\prime}，\quad \mathbf{w}(e)=\frac{1}{n} \sum_{i=1}^n \hat{w}_{e, i}^{\prime}
$$

在这些 embedding 中，$\mathbf{e}_{sec}, \mathbf{s}(e)$ 封装了 $e$ 的全局信息，而 $\mathbf{v}(e), \mathbf{w}(e)$ 包含了局部模态信息。

对于每个实体 $e$，我们可以为对比学习收集其候选集 $\mathcal{C}(e)=\left\{e_{sec}, \mathbf{s}(e), \mathbf{v}(e), \mathbf{w}(e)\right\}$，包括其全局和局部特征。其中 $\mathbf{e}^{\prime} \in \mathcal{C}(e)$ 被视为正样本。然后，我们使用批内负采样构建负样本对，并使用 InfoNCE（van den Oord 等人，2018）作为对比学习的基础。最终的 FBCL 目标可以表示为：

$$
\mathscr{L}_{\text {con }}=-\sum_{i=1}^{\mathcal{B}} \sum_{e_i^{\prime} \in \mathcal{C}\left(e_i\right)} \log \frac{\exp \left(\boldsymbol{cos}\left(\mathbf{e}_i, \mathbf{e}_i^{\prime}\right) / \tau\right)}{\sum_{j=1}^{\mathcal{B}} \exp \left(\boldsymbol{\operatorname {cos}}\left(\mathbf{e}_i, \mathbf{e}_j^{\prime}\right) / \tau\right)}
$$

其中 $\mathcal{B}$ 是批次大小，$\boldsymbol{\operatorname {cos}}(\cdot, \cdot)$ 是两个 embedding 的余弦相似度，$\tau$ 是温度超参数。通过这样的 FGCL 过程，MYGO 显著提高了其辨识各种实体中详细多模态属性的能力，从而提升了模型在 MMKGC 任务中的表现。最终，我们框架的整体训练目标可以表示为：

$$
\mathscr{L}=\mathscr{L}_{\text {kgc }}+\lambda \mathscr{L}_{\text {con }}
$$

其中 $\lambda$ 是一个超参数，用来控制对比损失 $\mathscr{L}_{\text {con }}$ 的权重。
