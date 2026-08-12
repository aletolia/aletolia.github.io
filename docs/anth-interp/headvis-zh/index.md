# HeadVis：注意力头可视化

*May 2026 · 原文: https://transformer-circuits.pub/2026/headvis/index.html*

---

### 引言

我们介绍 HeadVis——一个用于研究大语言模型中注意力头的交互式工具。在先前的工作中，可视化单个计算单元在整个数据分布上的激活方式已被证明十分有用——例如，[特征可视化](https://transformer-circuits.pub/2023/monosemantic-features/index.html#setup-interface) 一直是研究残差流特征、MLP 特征与 MLP 神经元的重要工具。HeadVis 在概念上与之类似，但针对的是注意力头。然而，注意力头比神经元更难可视化，因为它们秩较高、组合了两种不同的电路，并且在整个上下文窗口上激活。我们呈现几个使用 HeadVis 解读 Claude Haiku 3.5 注意力头的案例研究。我们还开源了 [HeadVis](https://github.com/anthropics/headvis) 的代码，并展示了 HeadVis 在 [Gemma 3](./gemma3/index.html) 的注意力头以及 [Haiku 3.5](./haiku/index.html) 的部分注意力头上的演示。

先前的工作表明，注意力头在广泛数据分布上的行为，与其在特定数据集上的行为可能截然不同。例如：

- \cite{gould2024successor} 在自定义数据集上发现了一个实现“后继”（在序数序列中预测下一个元素）的注意力头。当他们在更通用的数据集上分析该头时，注意到包括复制、比较和首字母缩略词生成在内的不同行为。

- \cite{lieberum2023does} 发现，一个负责复制选择题答案的注意力头还会枚举列表中的元素。
- \cite{mcdougall2023copy} 发现，IOI 电路中的一个抑制头广泛地执行“反归纳”或复制抑制。

我们构建 HeadVis，是为了借助注意力模式与注意力头输出的可视化、定量分布指标、低秩分量投影，以及经由 QK 与 OV 电路的稀疏自编码器（sparse autoencoder, SAE）特征归因，轻松地生成并检验关于注意力头功能的假设。

在本文中，我们：

- 通过研究一个简单的归纳头来介绍 HeadVis。
- 发现我们[先前研究](https://transformer-circuits.pub/2025/linebreaks/index.html#illusion) 的“行宽”头之一，在全分布上具有多种不同的行为。
- 表明我们[先前研究](https://transformer-circuits.pub/2025/attention-qk/index.html) 的“答案选择”行为，在远比选择题更广泛的数据分布上，复用了同样的“选择”机制。
- 找到一个抑制成对相关实体（如一个国家与其自身的城市）之间注意力的头。
- 给出我们对解读注意力的理论障碍的最新看法。

先前的工作与我们自己的工作共同浮现出一个主题：一个头在全分布上的行为很少与狭窄任务所暗示的一致，而这种差异并不总是容易刻画。

### 模糊归纳

我们从研究归纳入手，以熟悉 HeadVis 的用法。归纳头[^1] 以及更“模糊”的归纳变体（例如基于语言翻译、语义关系、更长上下文与多词元前缀的归纳）已在先前工作中得到广泛研究 \cite{ren2024identifyingsemanticinductionheads,akyurek2024incontextlanguagelearningarchitectures,crosbie2025inductionheadsessentialmechanism,wu2024retrievalheadmechanisticallyexplains}。HeadVis 浮现出若干归纳头，并让我们看到，“模糊”归纳如何能被理解为在特征基上执行归纳。

HeadVis UI 的第一个组件是一张头的散点图，两个坐标轴与点的颜色都可配置为任意逐头量，例如其层数、头索引或若干计算指标之一。位于某一指标极端的头，往往正是做着可解释事情的头，因此这提供了一种快速找到有趣头的简单方法。

*[**图 1：** HeadVis 的头选择器。我们实现了多种指标，以便轻松找到感兴趣的注意力头。点击任一坐标轴或色条上的选择器，即可更改你所可视化的指标。]*

我们可以借助一个名为归纳分数（induction score）[^2] 的代理指标来找到归纳头。我们发现，浅层归纳头往往执行字面复制，而中层归纳头则执行更模糊的归纳形式（即不直接作用于词元的归纳）。

*[**图 2：** 归纳头存在于模型的各个层中，但较深层中的归纳头开始实现更模糊的归纳形式。归纳分数仍然有用，但无法追踪模糊归纳，因为它只在词元空间中匹配前缀。]*

下面是该模糊归纳头在一个最高激活数据集示例上的注意力模式。这是 HeadVis 的核心 UI：按最大注意力模式分桶的注意力模式。乍看之下，我们注意到一些标准归纳的情形。将鼠标悬停在 `I’m` 中的 `‘m` 上，可以看到注意力指向上下文中更早处的 `I’m` 之后的词元。我们还注意到 `gleam` 具有最高的最大注意力模式，但上下文中并没有 `gleam` 的先前实例，因此这不可能是标准归纳。将鼠标悬停在 `gleam` 的任一词元上，可以看到注意力指向 “dream” 之后的换行符。不妨先猜一猜各个词元会回溯注意哪里，再悬停上去验证。我们发现过这样的模糊归纳头：它们匹配跨语言的单词、反义词、跨结构位置（一个列表的第一项与另一个列表的第一项相匹配）以及亲属称谓（aunt 与 uncle 相匹配）。

*[**图 3：** 模糊诗歌归纳头的一个最高激活数据集示例。每个词元按其在作为查询时、对上下文中所有键取最大值得到的注意力模式值着色。将鼠标悬停在一个词元上，可查看它注意哪些键。按 `t` 键可改为按词元作为键时的最大注意力模式着色。]*

为了理解是什么让这种归纳变得模糊，我们从 `gleam` 的第二个词元出发、回溯到 `dream` 之后的换行符，运行 QK 与 OV 归因 \cite{anthropic2025qkattributions}。QK 归因将注意力分数写成（查询特征，键特征）交互之和。在复制型归纳头中，排名靠前的特征交互应当是查询侧的 `current token is X` 与键侧的 `token after X`。在本例中，我们看到查询侧的 `-ea sound at line end` 与键侧的 `newline after an “ee” sound` 交互。这与复制型归纳的情形类似，只不过它匹配的不是特定词元，而是 `”ee” sound`。

OV 归因将头的输出写成（值特征，输出特征）交互之和：值特征位于被注意的词元处，输出特征位于查询词元处。此处排名最高的 OV 项与字面归纳完全相同：值侧的 `current token is \n` 与输出侧的 `say \n`。模糊性完全存在于 QK 之中；一旦头决定了注意何处，它就像字面归纳头一样进行复制。图 4 将这四种特征布局成一条电路；完整的 QK 与 OV 归因表位于 HeadVis UI 中。

*[**图 4：** 上述模糊诗歌归纳头的 QK 与 OV 电路图。所示特征在该头中具有最高的边际 QK 与 OV 归因。]*

QK 与 OV 归因是 HeadVis 的一项核心能力，可以为任意（查询，键）对实时生成；这些特征来自 \cite{anthropic2025qkattributions} 所用的同一个 1000 万特征[弱因果跨层转码器](https://transformer-circuits.pub/2024/crosscoders/index.html)（weakly causal crosscoder, WCC）。

这是 HeadVis 的典型工作流：挑一个在某项指标上处于极端的头，浏览它在数据集示例上的注意力模式，再用 QK 与 OV 归因来理解正在发生的事情。归纳头是一个工作流顺畅进行的干净案例。

### 一个多语义的注意力头

在 HeadVis 中可视化 \cite{anthropic2025linebreaks} 的注意力头时，我们注意到其中一个行宽头——一个通过统计自上一个换行符以来的字符数来计算行宽的浅层头——似乎不止做这一件事。从它在数据集示例上的注意力模式中，我们识别出三种不同的行为：

- 年份：分词器将 1000–1999 年的年份拆成两个词元，该头在它们之间注意。例如，`1776` 变成 `1` + `776`，该头从 `776` 注意 `1`。

- 多词元单词：分词器将长单词拆成多个词元，该头经常在它们之间注意。例如，`interpretability` 变成 `interpret` + `ability`，该头从 `ability` 注意 `interpret`。

- 换行符：该头从一个换行符词元注意到一个先前的换行符词元。这是先前研究过的行为——在这种情况下，该头借助前一层中字符计数头的帮助计算当前行的宽度。

*[**图 5：** 行宽头在其上激活的每种行为的示例序列。]*

我们通过检查哪些词元注意哪些词元来识别这些行为。HeadVis 为每个头预计算一个 PCA 视图，作为判断这类词元级区分是否也体现在该头 Q、K、O 激活中的廉价初步检查——外观不同的词元可能产生相似的激活，在这种情况下，看似三种行为，对该头而言可能只是一种。我们取注意力输出范数最高的（查询，键）词元对，收集该头在这些词元对上的 Q、K、O 激活[^3]，并对每组激活运行 PCA。如果前几个分量分离成簇，就提示该头具有值得分别研究的不同行为。我们将 PCA 视为假设生成工具而非检验：PCA 只把激活投影到一个捕获激活方差的低维子空间上，但不同的注意力行为可能在方差高的维度上看似可分，却并不真正影响注意力分数，甚至可能只有在更高维空间中才可分。原则上，一个头可以被构造得让 PCA 具有最大的误导性；在实践中，它一直是一个有用的起点。

对于这个头，三个 PCA 都在前几个主分量中把行为分离成不同的簇。

*[**图 6：** 该头在注意力输出范数较高的词元对上的 Q、K、O 激活的 PCA。颜色由针对每种行为硬编码的词元级规则分配（例如，年份：键=`1`，查询为 3 位数字，距离 1），与 PCA 无关。]*

Q、K、O 上的干净分离——连同词元级模式——让我们相当确信这个头是多语义的：一个头实现了若干互不相关的行为。在 HeadVis 中浏览更多数据集示例后，这三种行为解释了我们所见的大部分情况；几乎没有我们无法归类为其中之一的示例。对多语义性更直接的检验，是把该头拆分成若干单独的注意力头，每个头隔离一种行为，并表明它们合在一起能复现原头；我们将在讨论部分回到这一点。

知道一种行为存在，并不等于完全刻画了这种行为。对于年份行为，单个示例上的 QK 归因将一个“当前词元是 1”的键特征与一个“当前词元是 [3 位后缀]”的查询特征配对——这与注意力模式所暗示的一致，也与该行为覆盖全部 1000–1999 年相一致。但逐示例归因并没有给出能在任意输入上预测电路的规则：在固定提示中扫过各个年份时，该头只在 1000–1986 年间强注意，而这一约束无论是注意力模式还是归因都没有揭示出来。

*[**图 7：** 该头只在 1000–1986 年间强注意。]*

这是引言所描述的两种结果之一：一个因某项任务而被研究的头，结果在全分布上具有若干互不相关的行为。这差不多是我们所找到的最干净的示例——一个浅层头，因行为分离得干净而被选中，三种行为从词元模式中可见、经 PCA 证实，归因也大多讲得通。我们接下来研究的答案选择头是另一种结果，它不具备上述任何优势。

### 答案选择

我们更仔细地研究了 \cite{anthropic2025qkattributions} 中的答案选择头。在本例中，回答选择题也只是该头的一种特定行为，但我们看到的其他行为似乎与之密切相关！我们发现，“答案选择”更应被描述为一种更一般的头行为：通过识别即将被提及的实体，帮助模型复述先前的上下文。

我们从同一个注意力模式视图入手，正是这个视图浮现出行宽头的三种行为。激活最强的示例中没有一个是选择题。这并不令人意外，因为选择题只占完整数据集的很小一部分，模型不太可能把整个头都专门用于它。

*[**图 8：** HeadVis 中答案选择头的几个最高激活示例。]*

注意力模式只让人对头在做些什么有模糊的认识。我们发现这是中层头的典型特征：词元级模式不如浅层头中那样可读。应用分离出行宽头行为的同一个 PCA 视图，我们得到不同的结果：Q、K、O 激活形成一个没有簇的连续云团。

*[**图 9：** 答案选择头的 Q、K、O 激活的 PCA，与上文图 6 类似。]*

在这个头中，PCA 没有让激活形成聚类，因此它对于理解该头是否实现多种行为没有帮助。我们转而求助于 QK 归因，我们先前的工作已经绘制出选择题情形的图谱：一个“即将说出选择题答案”的查询特征与一个“正确答案”的键特征交互，OV 电路则将答案标签复制到输出。对来自全分布的示例运行 QK 与 OV 归因后，一种模式浮现出来。OV 电路总是做同一件事：复制存储在被注意词元处的内容，与归纳颇为相似。QK 电路才是复杂之所在：跨示例来看，它在键侧匹配“模型已标记为相关的内容”特征，在查询侧匹配“即将产生与该标记相匹配的内容”特征。

*[**图 10：** 一道选择题的 QK 电路。]*

*[**图 11：** 一个情感分类函数的 QK 电路。]*

*[**图 12：** 两个说话者之间一段对话的 QK 电路。]*

这些示例看起来结构相似，但每个示例上激活的特征集合互不相交——例如，选择题查询特征从不与情感键特征同时激活。这引出一个问题：该头究竟是实现了一种行为，还是实现了许多共享同一模板的行为。

遵循 \cite{ameisen2025circuit} 的做法，我们通过该头 QK 电路中特征之间的[虚拟权重](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#global-weights) 来探究这一点。我们将“正确答案”特征投影经过该头的键权重，并询问在投影空间中还有哪些其他特征落在其附近。在过完备特征基中研究头行为，或许能向我们展示低秩投影（如 PCA）无法发现的键与查询之间的关系。

对齐最强的那些特征并非专门与选择题有关，而是“这是相关内容”的其他实例——这些特征在残差流中近似正交，而该头的 K 投影将它们映射到彼此接近的位置。这些特征激活的语境远比选择题广泛。查询侧也是如此。

*[**图 13：** 经过该头的键或查询权重投影后，与“正确答案”特征余弦相似度最高的那些特征。]*

作为对照，对线宽头的年份行为做同样的分析，结果正好相反：与某个年份特征虚拟权重最高的特征全都与年份有关。

我们最好的猜测是，这个头是单语义的，但我们并不完全确信。上述归因与虚拟权重只覆盖了少量示例。它们看起来一致，但我们还没有系统性地调查过这个头。更深层的问题在于，我们还没有一个足够清晰的行为假设可供验证。如果我们能用文字描述这个头的功能，就可以拿数据集中的每个示例去检验它，并在出现不符时发现（或让一个评分模型发现）异常。没有这样的描述，我们只能看着一个小样本，观察到目前为止一切都符合同一个宽泛模板。

### 同集合抑制

我们以一个只部分理解的中间层头来结束案例分析。我们在它的 QK 几何结构中分离出了一种机制——足以预测它在受控输入上的行为——然后追踪了为什么这种机制在真实数据上难以察觉。

在 HeadVis 中，你可以依据该头对各词元的平均注意力，查看它排名靠前的查询与键词元。对这个头来说，两个列表都以地点为主，夹杂一些人名和队名。

*[**图 14：** 按平均注意力排序的排名靠前的查询与键词元。]*

数据集示例显示，该头在地点与人之间建立注意力联系，但仅凭注意力模式，很难再看出更多东西。在下面第一个序列中，从 `Ivory`（在“Ivory Coast”中）它注意到“we've had `a` real laugh”中一个无关的 `a`。在第二个序列中，该头从 `Knicks` 注意到 `Kidd`，却没有从 `Nets` 注意到 `Kidd`——尽管 Jason Kidd 最出名的身份正是篮网（Nets）球员。

*[**图 15：** 两个数据集示例。]*

第二个示例暗示，那种显而易见的关联——球队关注自己的球员——并不成立。如果你为 HeadVis 配置了服务器，就可以输入任意序列，查看该头在其上的注意力[^4]。我们借此构造了一些句子，每句含有一个属于、一个不属于的项目。从 `Madrid` 出发，该头注意到的是 `Messi` 而不是 `Modrić`，尽管 Modrić 效力于皇家马德里，而 Messi 并没有。从 `Europe` 出发，该头注意到的是 `Zambia`——列表中唯一一个非欧洲国家。

*[**图 16：** 两个自定义序列。]*

为了系统地检验这一点，我们生成了 185 个形如“{五个城市} 都是 {国家} 的城市”的句子，其中四个城市属于该国，一个是外国城市。在 98% 的情况下，国家词元对外国城市的注意力都高于对其本国任何一个城市的注意力。

我们发现这种关系明确地体现在 QK 电路的几何结构中。对每个国家，我们计算该头在“The country {X}”上的查询向量，减去所有国家的均值，并对“The city {X} is well known”上的城市键向量做同样的处理。一个国家的查询向量与其本国城市的键向量之间的余弦相似度强烈为负（均值 −0.47），与别处城市的则接近为零（均值 +0.01）。W_Q 和 W_K 投影造就了这一几何结构：在残差流中，一个国家与其本国城市具有正的余弦相似度（均值 +0.17），因此该头将符号翻转了过来。

*[**图 17：** 去均值后的国家查询向量与城市键向量之间的余弦相似度。深色对角线：每个国家的查询与其本国城市的键反对齐；非对角线元素接近零。]*

这一几何结构与自定义句子上 98% 的结果共同证明，同集合抑制是真实存在的。但带着这一模式回到 HeadVis 中该头的数据集示例，情况并不干净。对于前文那个多语义头，一旦我们知道了那三种行为，就能在一个又一个示例中把它们挑出来。而在这里，有些示例根本不涉及地点；有些示例中的注意力发生在同一地区的多个地点之间，没有异类；对很多示例，我们得仔细端详才能判断它们是否符合。我们认为，我们分离出来的只是这个头在处理地点时行为的一部分——它不像多语义头的那些行为那样能与其余部分干净地分开，也没有完整解释那些地点示例。

我们回到前面提到的 `Ivory`→`a` 示例，查询位置上是一个地点，但键位置上并没有相关词元。从 Ivory 到 `a` 的 QK 归因显示，交互最强的特征对是键侧的赞比亚（Zambian）特征与查询侧的科特迪瓦（Ivory Coast）特征，它们对注意力分数有正向贡献。

*[**图 18：** 从 `Ivory` 到 `a` 的 QK 归因。最强的特征对是查询侧的科特迪瓦特征与键侧的赞比亚特征。]*

尽管我们研究的这条注意力边是从 `Ivory` 到一个随机词元，但序列的其余部分看起来与我们的国家-城市结果相似。这个序列讲的是去赞比亚的一次旅行——序列前面提到的 [chikumbuso](https://www.chikumbuso.com/) 是一家赞比亚非营利组织——而 Ivory Coast 是唯一提到的另一个国家。该头从这个突兀的国家注意到 `a`，部分原因是一个赞比亚特征，这与“注意力发生在匹配之物与不匹配之物之间”如出一辙。

为了检验这一点，我们把序列中的 `Ivory Coast` 替换为 `Zambia`，对 `a` 的注意力从 0.87 下降到 0.08；替换成赞比亚首都 `Lusaka` 时也是如此（0.10）。换用其他 51 个非洲国家时，注意力大多远高于这个水平（中位数 0.27）。因此，与我们那些狭窄的提示一样，查询位置上是别的国家时会建立注意力，是同一个国家时则不会[^5]。

*[**图 19：** 在所有非洲国家中，被替换进来的国家词元对 `a` 的注意力。赞比亚受到抑制；大多数其他国家都会关注 `a`。]*

剩下的问题是，一个赞比亚特征怎么会激活在 `a` 上。在 HeadVis 中，你可以选取序列上的任意（查询，键）对，按所有头在该对上的注意力模式对它们排序[^6]。`chikumbuso` 是序列前面唯一与赞比亚相关的词元，于是我们寻找一个在 `a` 与 `chikumbuso` 之间建立注意力的头，并检查了它的 OV 电路。那个头写入的，正是我们这个头的键侧所读取的赞比亚特征！这就是 K 组合 \cite{elhage2021mathematical}：一个头向某个位置写入，后面的头从该位置读取。

*[**图 20：** 一个更早的头从 `a` 注意到 `chikumbuso`，并写入这个头所读取的赞比亚特征。]*

这个例子部分解释了为什么该头在数据集上的注意力模式不易解读：有时某个位置上的关键，是那里激活的特征，而不是词元本身。这里，该头注意到 `a`，是因为更早的一个头在那里写入了一个赞比亚特征；仅凭词元无从看出这一点。

这四个头展示了这套工作流能够带你走多远：归纳案例端到端干净利落地跑通；线宽头的三种行为从词元模式中即可看出，并经 PCA 确认；答案选择头需要特征层面的分析才能形成假设；而在这里，受控输入与 QK 几何结构恢复出了一种我们从数据集示例中看不到的机制。本节每一步都使用了 HeadVis 的不同视图——用词元排行榜看什么占主导，用自定义序列检验猜测，用 QK 归因解读令人困惑的示例，用（查询，键）对上的头排序追踪组合。我们开源了这个工具，以便其他人也能在自己的模型上开展这些研究。

### 开源 HeadVis

HeadVis 由三部分组成：一个用于浏览和交互式探查头的前端、一个在数据集上预计算每个头的指标与注意力模式的离线脚本，以及一个处理 QK 与 OV 归因等实时查询的服务器。我们开源了前端，以及它与两个后端之间接口的规范。你可以在[此处](https://github.com/anthropics/headvis)找到该仓库，以及将前端连接到你自己模型的说明。我们建议让 Claude 直接阅读该仓库，为你的环境实现后端部分。一个托管在 Haiku 3.5 部分头上的演示在[这里](./haiku/index.html)[^7]。覆盖 Gemma 3 1B \cite{gemma3} 全部头的 HeadVis 在[这里](./gemma3/index.html)。

我们在内部发现的一些有用功能没有包含在这次开源发布中：

- 在自定义提示上实时运行的能力
- 理解头输出向量的各种方法，例如 logit 透镜（logit lens）
- 报告各序列上头的注意力模式之间相似度的指标，用于寻找功能相似的头
- 头组合热力图（即该头的 OV 权重与后面某个头的 Q/K 权重共享子空间）

可能还有更多扩展能让这个工具更有用。

### 相关工作

交互式注意力头可视化。\cite{vig2019bertviz} 试图在单个提示内理解 BERT 的注意力头，并提供神经元层面的注意力模式解释。\cite{hoover2020exbert} 在此基础上增加了语料库级检索，让用户能将人工指定的输入与带标注的参考集进行匹配。\cite{yeh2023attentionviz} 与 HeadVis 最为接近：它将来自大量数据集示例的查询和键向量联合嵌入到一个共享的低维空间中，类似于我们的 PCA 图，并跨序列聚合头的行为。\cite{mossing2024tdb} 则走向互补的方向：从提示和行为出发，识别出哪些头、神经元和 SAE 潜在特征有贡献，并为其浮现出的组件提供数据集示例页面。

HeadVis 受这些工作的启发，但在两方面做了延伸：既要理解单个头在整个数据分布上的运作方式，也要开发出能简化复杂注意力行为理解的可扩展工具。例如，如果没有将 HeadVis 扩展为支持 QK/OV 特征归因与头组合可视化，我们要理解答案选择头的整体行为、或同集合抑制头与其他头的组合，就会困难得多。

注意力头行为编目。许多先前的工作刻画了单个注意力头的行为；综述见 \cite{zheng2024attention}。以下是一些例子：

- 单一功能。\cite{voita2019analyzing,clark2019what,kovaleva2019revealing} 用离散角色描述每个注意力头——句法追踪、关注稀有词元、或关注特定的位置偏移。任务电路分析（如 \cite{wang2023interpretability}）识别出组合起来执行单一行为的专门头集合。\cite{wu2024retrievalheadmechanisticallyexplains} 通过一个合成复制任务识别出一类稀疏而普遍的“检索头”。\cite{conmy2023towards} 和 \cite{ferrando2024information} 等自动化工具试图扩展这种任务电路方法，以算法方式找出单一的注意力机制。
- 多语义性。一条互补的研究脉络观察到，单个头常常在整个数据分布中扮演多种角色。\cite{gould2024successor} 在 Pythia-1.4B 的单个头中发现了继任（successorship）、缩写续写、复制和大于比较等行为。\cite{merullo2024circuit} 发现有些头同时参与多个任务电路，例如 IOI 电路与 Colored Objects 电路之间约 78% 的头重叠。\cite{kissane2024interpreting} 使用 Attention-output SAE 发现，GPT-2 中几乎所有头都参与不止一种可解释的行为。
- 一般行为。\cite{mcdougall2023copy} 发现，先前被归因于 GPT-2 的 L10H7 的若干行为，其实是复制抑制（copy suppression）这一单一一般行为的各个侧面。\cite{lieberum2023does} 对 Chinchilla 的选择题头应用基于 SVD 的分解，找出了只部分执行选择题作答的子空间，这与我们发现答案选择头的过程类似。\cite{ren2024identifyingsemanticinductionheads} 将归纳从字面词元匹配推广到语义关系，是我们在模糊归纳头中观察到的特征空间匹配的先前工作。

我们把 HeadVis 视为一种用于发现更多这类注意力行为与病理现象的工具。我们预期，理解注意力头生物学的个别实例将持续带来深刻洞见，并指引我们在彻底分解注意力机制上的未来进展，例如借助 \cite{he2025lorsa} 或 \cite{jermyn2025attention} 的方法。

### 讨论

将注意力研究中的开放问题与 MLP 对照着看是很有用的。MLP 层由许多处于叠加中的多语义神经元组成。对于 MLP，我们可以借助转码器（transcoder）学到单语义、稀疏的特征，忠实地重建该层的大部分计算[^8]。

我们还不知道如何对注意力层做类似的事情。目标在精神上是相同的：学到单语义、稀疏的“注意力特征”，忠实地重建一个注意力层。我们预期，一个注意力特征看起来大致像一个注意力头——也许带有修改过的非线性或不同的头维度——但足够接近，以至于用标准头作为心理模型是合理的[^9]。如果我们有了这样的分解，我们猜想：线宽头中的三种行为会表现为三个独立的特征，而答案选择头会表现为一个单一的高秩特征[^10]。

有四个障碍挡在前面：

- 注意力生物学——我们不知道应当期待什么样的注意力特征，这让我们难以判断一种分解方法是否奏效。
- 头的多语义性——注意力头可以在没有叠加的情况下就是多语义的，而 MLP 神经元不可能如此。
- 注意力叠加——注意力特征很可能分散在多个头之间，但与 MLP 叠加不同，我们在真实 LLM 中还没有具体的例子。
- 高秩注意力特征——某些注意力特征可能是高秩的，这使它们远比秩为 1 的 MLP 特征难以解释。

##### 注意力生物学

我们还不知道注意力特征总体上是什么样子，因此我们研究单个头作为替代。头很可能比注意力特征更难解释——一个头可能是多语义的，也可能处于叠加之中——但 HeadVis 足以让我们从中学到东西。本文中的两个头给了我们分解方法需要处理的具象例子：线宽头展示了一个头干净利落地实现若干互不相关的行为，答案选择头则指向一个可能不可约地高秩的单语义特征。虽然我们还没有完全理解这些机制，但知道它们确实存在，就是迈向刻画它们所属的注意力特征的非常有用的一步。

我们很期待看到 HeadVis 在长上下文方面的工作。特别是，文献中对频繁进行长距离注意的头研究得较少。理解注意力在长对话记录上如何运作，对于研究模型的高级能力（例如系统提示在整段对话中引导 Assistant 所起的作用）可能大有裨益。

我们还想知道，头在多大程度上会利用 softmax 竞争来实现优先级逻辑。一个头只需让 `A` 的键得分高于 `B` 的键，就能实现“注意 `A`（若其存在），否则注意 `B`”。这是注意力机制的基本组成部分吗？到目前为止，我们一次只研究单个注意力边（单个键和查询词元），因此任何依赖于键之间竞争的逻辑对我们来说都是不可见的。

还有更多注意力“生物学”有待发现，尽管目前尚不清楚是否必须先解决下面这些障碍。无论如何，我们都鼓励每一位研究注意力的人，在开源模型上花几个小时用一用 HeadVis。有些直觉只有亲眼观察才能获得，而且往往几个小时就足以找到一个能清晰体现你想研究的理论问题的头。

##### 头的多语义性

我们的线宽头（line width head）让这个障碍显得触手可及。多语义性的存在此前已知，但我们的例子异常简单：一个早期层的头，其中三种行为可以通过 PCA 分离，而且它的注意力模式本身就可直接解读。我们没费什么功夫就找到了它，并且预计还会有更多类似的头。这类头是分解方法的天然测试用例。

我们还想把一个理论要点讲清楚：注意力头可以在没有叠加的情况下就是多语义的。头的输出是高秩的，因此它可以把互不相关的行为写入可区分的方向，下游组件仅凭这个头就能读出这些行为。MLP 神经元做不到这一点——它的输出只有一个方向，因此一个多语义的神经元需要其他神经元来消除其含义的歧义，而这按定义就是叠加。因此，头的多语义性有两种形态：有叠加的和没有叠加的。我们尚未确定线宽头属于哪一种，但分解方法必须同时应对这两种。

我们很期待看到某个多语义的头被拆分成单语义的单元。以线宽头这样的头为例：为三个注意力头找到各自隔离出一种行为的权重，然后检验它们合在一起能否复现原头的输出。更远大的目标，是设计出一种能对任意头自动完成这一拆分的流程[^11]。

##### 注意力叠加

我们怀疑有些注意力特征分布在多个头上，但尚未在真实 LLM 中确认到这样的例子。我们很期待看到一个实例：一种行为，单独看任何一个头都难以解释，但当你把几个共同在词元之间进行注意的头放在一起看时，它就浮现出来了[^12]。我们也很期待看到 \cite{bricken2023monosemanticity} 中希伯来特征（Hebrew feature）在注意力上的对应物：一个隐藏在多个头的低激活值中的特征，而其中每个头在高激活值下看起来都各自可解释。在此基础上，我们还想看到一个能复现该例现象学（phenomenology）的玩具模型。过去的玩具模型——包括我们自己的 \cite{jermyn2023attention}——确实产生了某种形式的注意力叠加，但尚不清楚它们是否捕捉到了真实头中实际发生的现象。一个得到确认的真实世界例子，会给玩具模型一个可以复现的对象。最远大的目标，则是一个能自动消除叠加的流程。

##### 解释高秩注意力特征

多语义性和叠加都说明，单个头不是正确的分析单元；解决它们意味着要找到更好的单元。这个障碍则有所不同：即使单元选对了，它也可能是高秩的，而该领域在解释高秩对象方面经验甚少。

MLP 层有两个特性，使它们更容易研究：

(i) 它们由神经元组成，我们可以构建一个架构完全相同的转码器（transcoder）来研究它们。我们相信，我们学到的特征可以[在机制上忠实](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#limitations-faithfulness)于原始 MLP。[^13]

(ii) 每个转码器神经元就是一个特征，且是秩为 1 的对象，易于解释和推理。

而就注意力而言，层的单元并不是秩为 1 的：一个头的 softmax 前得分是两个位置之间的一个秩为 $d_{\text{head}}$ 的双线性形式。我们怀疑，若干秩为 1 的头之和[^14]无法很好地逼近每一个真实头，因此要达成 (i)，很可能需要秩大于 1 的部件。大多少尚不清楚——也许大多数注意力特征都落在秩 2–6 的范围内，是可以处理的，因为先前的工作已经解释过大致这个范围的特征 \cite{engels2025languagemodelfeaturesonedimensionally,lieberum2023does,anthropic2025linebreaks}；又或许有些特征确实需要接近 $d_{\text{head}}$ 的秩，那就既难以可视化，又难以推理。

我们很期待看到某个高秩注意力特征被完整地解析出来：找到一个很可能是单语义的、且未处于叠加中的头，并对它在整个数据分布上都成立的计算给出描述。答案选择头（answer selection head）或许是一个候选，但我们离完整描述它还差得很远，也不知道它是否涉及叠加。

##### 现有的分解方法

已经有几种方法在尝试我们所说的这种分解。注意力输出 SAE（attention-output SAE）\cite{kissane2024interpreting}、多词元转码器（multi-token transcoder）\cite{jermyn2025attention} 和低秩稀疏注意力（Low-Rank Sparse Attention）\cite{he2025lorsa,shu2026crm} 都会为注意力层学习一组稀疏、过完备的秩为 1 的 OV 方向基。它们的区别在于如何处理注意力模式——注意力输出 SAE 位于基础模型注意力之后，多词元转码器混合使用基础模型冻结的模式，而 Lorsa 学习新的、由特征组共享的全秩 QK 电路——但三者的 OV 侧是相同的。这些方法所针对的正是多语义性和叠加：多语义的头会把不同行为写入不同的输出方向，而一个分布在多个头上的特征仍然只写入一个净方向，因此秩为 1 的 OV 基原则上可以同时隔离这两者。

秩为 1 的 OV 设计带来的一个后果是：具有更高秩 OV 的特征——例如继任（succession）或复制归纳（copying induction）——会被拆分到许多学到的特征上，而它们本可以被更好地理解为单一行为。这些方法还继承了高秩可解释性这一挑战，因为它们的 QK 电路仍然是高秩的。我们很期待看到这些分解方法被应用到少数几个能用 HeadVis 理解的早期层头上；这很可能是定性地了解其性能的最简单途径。

##### 结论

我们对解释注意力持乐观态度。HeadVis 让大多数障碍都有了具体的例子，而且一旦工具存在，这些例子并不难找到。既然有了一套清晰的例子可供着手，我们认为该领域已经准备好直接攻克这些障碍。这需要新的方法，而这些例子会告诉我们新方法需要应对什么。

## 脚注

[^1]: 给定模式 `[A][B] ... [A] -> [B]`，头会从第二个 `A` 回看 `B`，以预测其再次出现。A 只需模糊匹配即可。

[^2]: 归纳分数（induction score）是一个头在对应于归纳的字面复制情形的（查询、键）词元对上的平均注意力模式。在一个形如 `[A][B]…[A]->[B]` 的序列中，归纳头会从第二个 A 注意第一个 B。这与 \cite{olsson2022incontext} 中的前缀分数（prefix score）相同，区别仅在于前缀分数是在由重复随机字符串构成的合成数据集上计算的，而归纳分数是在标准数据集上计算的。

[^3]: 一个（查询、键）词元对的 O 激活，是键的残差流经过 OV 电路、再按注意力模式缩放后的结果。我们略去 V，因为它与 O 之间只差固定的线性映射 W_O 和注意力模式，因此它的 PCA 在定性上是相同的。

[^4]: 自定义序列输入未包含在开源版本中；参见 Open Source HeadVis。

[^5]: 两种设置有一处不同：在自定义句子中，国家注意到的是不匹配的城市；而这里，头是从不匹配的国家出发进行注意的。我们对此并不意外——自定义提示只隔离了更广泛行为中的一个狭窄情形，而不是唯一的情形。

[^6]: 该功能未包含在开源版本中；参见 Open Source HeadVis。

[^7]: 前端在查询服务器之前会先检查预计算的结果文件，因此静态部署可以服务任何预先生成的分析。托管演示正是利用这一点，在没有在线后端的情况下，为本文中的几个示例提供了缓存的 QK 和 OV 归因。

[^8]: 这种分解并不完美，但转码器特征通常在定性上比单个 MLP 神经元更可解释，而且在理解更广泛的模型计算（例如追踪完整电路）方面也很有价值。

[^9]: 注意力特征有可能看起来并不像一个注意力头，但当我们对注意力层任选分解方式时，就开始要面对[机制忠实性](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#limitations-faithfulness)方面的担忧了。

[^10]: 注意力叠加可能会让事情变得更复杂。线宽头中的每种行为都会是某个独立注意力特征的一部分，但这些注意力特征可能跨多个头实现。类似地，构成答案选择头的高秩特征也可能与其他注意力头处于叠加之中。

[^11]: 叠加仍可能参与其中，但我们认为，在忽略叠加的情况下研究多语义头的拆分，依然是有趣的。

[^12]: \cite{anthropic2025linebreaks} 中的断行头（line-breaking head）不算，因为即使它们合起来实现了一种复杂行为，每个头本身也仍然是可解释的。

[^13]: 请注意，这是一个假设，而且它有时[并不成立](https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html)。

[^14]: 所谓秩为 1 的头，我们指的是 d_head = 1，因此它的 QK 电路和 OV 电路都是秩为 1 的。

## 参考文献

- [gould2024successor]: Gould, Rhys, Ong, Euan, Ogden, George, Conmy, Arthur, “Successor Heads: Recurring, Interpretable Attention Heads In The Wild”, International Conference on Learning Representations, 2024
- [lieberum2023does]: Lieberum, Tom, Rahtz, Matthew, Kram{\'a}r, J{\'a}nos, Nanda, Neel, Irving, Geoffrey, Shah, Rohin, Mikulik, Vladimir, “Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla”, arXiv preprint arXiv:2307.09458, 2023
- [mcdougall2023copy]: McDougall, Callum, Conmy, Arthur, Rushing, Cody, McGrath, Thomas, Nanda, Neel, “Copy Suppression: Comprehensively Understanding an Attention Head”, arXiv preprint arXiv:2310.04625, 2023
- [ren2024identifyingsemanticinductionheads]: Jie Ren, Qipeng Guo, Hang Yan, Dongrui Liu, Quanshi Zhang, Xipeng Qiu, Dahua Lin, “Identifying Semantic Induction Heads to Understand In-Context Learning”, 2024
- [akyurek2024incontextlanguagelearningarchitectures]: Ekin Akyürek, Bailin Wang, Yoon Kim, Jacob Andreas, “In-Context Language Learning: Architectures and Algorithms”, 2024
- [crosbie2025inductionheadsessentialmechanism]: Joy Crosbie, Ekaterina Shutova, “Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning”, 2025
- [wu2024retrievalheadmechanisticallyexplains]: Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu, “Retrieval Head Mechanistically Explains Long-Context Factuality”, 2024
- [olsson2022incontext]: Olsson, Catherine, Elhage, Nelson, Nanda, Neel, Joseph, Nicholas, DasSarma, Nova, Henighan, Tom, Mann, Ben, Askell, Amanda, Bai, Yuntao, Chen, Anna, Conerly, Tom, Drain, Dawn, Ganguli, Deep, Hatfield-Dodds, Zac, Hernandez, Danny, Johnston, Scott, Jones, Andy, Kernion, Jackson, Lovitt, Liane, Ndousse, Kamal, Amodei, Dario, Brown, Tom, Clark, Jack, Kaplan, Jared, McCandlish, Sam, Olah, Chris, “In-context Learning and Induction Heads”, Transformer Circuits Thread, 2022
- [anthropic2025qkattributions]: Kamath, Harish, Ameisen, Emmanuel, Kauvar, Isaac, Luger, Rodrigo, Gurnee, Wes, Pearce, Adam, Zimmerman, Sam, Batson, Joshua, Conerly, Thomas, Olah, Chris, Lindsey, Jack, “Tracing Attention Computation Through Feature Interactions”, Transformer Circuits Thread, 2025
- [anthropic2025linebreaks]: Gurnee, Wes, Ameisen, Emmanuel, Kauvar, Isaac, Tarng, Julius, Pearce, Adam, Olah, Chris, Batson, Joshua, “When Models Manipulate Manifolds, The Geometry of a Counting Task”, Transformer Circuits Thread, 2025
- [ameisen2025circuit]: Ameisen, Emmanuel, Lindsey, Jack, Pearce, Adam, Gurnee, Wes, Turner, Nicholas L., Chen, Brian, Citro, Craig, Abrahams, David, Carter, Shan, Hosmer, Basil, Marcus, Jonathan, Sklar, Michael, Templeton, Adly, Bricken, Trenton, McDougall, Callum, Cunningham, Hoagy, Henighan, Thomas, Jermyn, Adam, Jones, Andy, Persic, Andrew, Qi, Zhenyi, Ben Thompson, T., Zimmerman, Sam, Rivoire, Kelley, Conerly, Thomas, Olah, Chris, Batson, Joshua, “Circuit Tracing: Revealing Computational Graphs in Language Models”, Transformer Circuits Thread, 2025
- [elhage2021mathematical]: Elhage, Nelson, Nanda, Neel, Olsson, Catherine, Henighan, Tom, Joseph, Nicholas, Mann, Ben, Askell, Amanda, Bai, Yuntao, Chen, Anna, Conerly, Tom, DasSarma, Nova, Drain, Dawn, Ganguli, Deep, Hatfield-Dodds, Zac, Hernandez, Danny, Jones, Andy, Kernion, Jackson, Lovitt, Liane, Ndousse, Kamal, Amodei, Dario, Brown, Tom, Clark, Jack, Kaplan, Jared, McCandlish, Sam, Olah, Chris, “A Mathematical Framework for Transformer Circuits”, Transformer Circuits Thread, 2021
- [gemma3]: Gemma Team, “Gemma 3 Technical Report”, 2025
- [vig2019bertviz]: Vig, Jesse, “A Multiscale Visualization of Attention in the Transformer Model”, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, 2019
- [hoover2020exbert]: Hoover, Benjamin, Strobelt, Hendrik, Gehrmann, Sebastian, “exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models”, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, 2020
- [yeh2023attentionviz]: Yeh, Catherine, Chen, Yida, Wu, Aoyu, Chen, Cynthia, Vi{\'e}gas, Fernanda, Wattenberg, Martin, “AttentionViz: A Global View of Transformer Attention”, IEEE Transactions on Visualization and Computer Graphics, 2023
- [mossing2024tdb]: Mossing, Dan, Bills, Steven, Tillman, Henk, Dupré la Tour, Tom, Cammarata, Nick, Gao, Leo, Achiam, Joshua, Yeh, Catherine, Leike, Jan, Wu, Jeff, Saunders, William, “Transformer Debugger”, \url{https://github.com/openai/transformer-debugger}, 2024
- [zheng2024attention]: Zheng, Zifan, Wang, Yezhaohui, Huang, Yuxin, Song, Shichao, Yang, Mingchuan, Tang, Bo, Xiong, Feiyu, Li, Zhiyu, “Attention Heads of Large Language Models: A Survey”, arXiv preprint arXiv:2409.03752, 2024
- [voita2019analyzing]: Voita, Elena, Talbot, David, Moiseev, Fedor, Sennrich, Rico, Titov, Ivan, “Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned”, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019
- [clark2019what]: Clark, Kevin, Khandelwal, Urvashi, Levy, Omer, Manning, Christopher D., “What Does BERT Look At? An Analysis of BERT's Attention”, Proceedings of the 2019 ACL Workshop BlackboxNLP, 2019
- [kovaleva2019revealing]: Kovaleva, Olga, Romanov, Alexey, Rogers, Anna, Rumshisky, Anna, “Revealing the Dark Secrets of BERT”, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019
- [wang2023interpretability]: Wang, Kevin Ro, Variengien, Alexandre, Conmy, Arthur, Shlegeris, Buck, Steinhardt, Jacob, “Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small”, International Conference on Learning Representations, 2023
- [conmy2023towards]: Conmy, Arthur, Mavor-Parker, Augustine N., Lynch, Aengus, Heimersheim, Stefan, Garriga-Alonso, Adri{\`a}, “Towards Automated Circuit Discovery for Mechanistic Interpretability”, Advances in Neural Information Processing Systems, 2023
- [ferrando2024information]: Ferrando, Javier, Voita, Elena, “Information Flow Routes: Automatically Interpreting Language Models at Scale”, arXiv preprint arXiv:2403.00824, 2024
- [merullo2024circuit]: Merullo, Jack, Eickhoff, Carsten, Pavlick, Ellie, “Circuit Component Reuse Across Tasks in Transformer Language Models”, International Conference on Learning Representations, 2024
- [kissane2024interpreting]: Kissane, Connor, Krzyzanowski, Robert, Bloom, Joseph Isaac, Conmy, Arthur, Nanda, Neel, “Interpreting Attention Layer Outputs with Sparse Autoencoders”, arXiv preprint arXiv:2406.17759, 2024
- [he2025lorsa]: He, Zhengfu, Wang, Junxuan, Lin, Rui, Ge, Xuyang, Shu, Wentao, Tang, Qiong, Zhang, Junping, “Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition”, arXiv preprint arXiv:2504.20938, 2025
- [jermyn2025attention]: Jermyn, Adam, Lindsey, Jack, Luger, Rodrigo, Turner, Nick, Bricken, Trenton, Pearce, Adam, McDougall, Callum, Thompson, Ben, Wu, Kai, Batson, Joshua, Rivoire, Olivier, Olah, Chris, “Progress on Attention”, Transformer Circuits Thread, 2025
- [bricken2023monosemanticity]: Bricken, Trenton, Templeton, Adly, Batson, Joshua, Chen, Brian, Jermyn, Adam, Conerly, Tom, Turner, Nick, Anil, Cem, Denison, Carson, Askell, Amanda, others, “Towards Monosemanticity: Decomposing Language Models With Dictionary Learning”, Transformer Circuits Thread, 2023
- [jermyn2023attention]: Jermyn, Adam, Olah, Chris, Henighan, Tom, “Attention Head Superposition”, Transformer Circuits Thread, 2023
- [engels2025languagemodelfeaturesonedimensionally]: Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, Max Tegmark, “Not All Language Model Features Are One-Dimensionally Linear”, 2025
- [shu2026crm]: Shu, Wentao, Ge, Xuyang, Zhou, Guancheng, Wang, Junxuan, Lin, Rui, Song, Zhaoxuan, Wu, Jiaxing, He, Zhengfu, Qiu, Xipeng, “Bridging the Attention Gap: Complete Replacement Models for Complete Circuit Tracing”, OpenMoss Interpretability Blog, 2026
