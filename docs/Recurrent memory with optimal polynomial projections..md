## 摘要:

在从序列数据中学习的中心问题是，随着更多数据被处理，以一种增量方式表示累积的历史记录。我们引入了一个通用框架 (HiPPO)，用于**==将连续信号和离散时间序列在线压缩到多项式基上==**。**<u>给定一个指定过去每个时间步骤重要性的测度，HiPPO 产生了一个自然的在线函数逼近问题的最优解</u>**。作为特例，我们的框架从第一原理简单推导出最近的 Legendre 记忆单元 (LMU)，并推广了诸如 GRU 等循环神经网络的通用门控机制。这个形式框架产生了一种新的记忆更新机制 (HiPPO-LegS)，可以随时间扩展以记住所有历史，避免了对时间尺度的先验假设。HiPPO-LegS 享有时间尺度稳健性、快速更新和有界梯度的理论优势。通过将记忆动力学纳入循环神经网络，HiPPO RNN 在经验上可以捕获复杂的时间依赖性。在基准置换 MNIST 数据集上，HiPPO-LegS 设置了 98.3% 的新的最佳精度记录。最后，在一个新的轨迹分类任务中测试对时间尺度外的分布和丢失数据的稳健性，HiPPO-LegS 的精度比 RNN 和神经 ODE 基线高出 25-40%。	

## 引言:

对序列数据进行建模和学习是现代机器学习中的一个基本问题，影响到诸如语言模型、语音识别、视频处理和强化学习等任务。建模长期和复杂时间依赖性的一个核心方面是记忆，或者存储和并入来自先前时间步骤的信息。挑战在于使用有限的存储空间学习整个累积历史的表示，这必须在接收到更多数据时在线更新。

一种成熟的方法是对随时间演化并吸收更多信息的状态进行建模。这种方法在深度学习中的体现是循环神经网络 (RNN)，人们已经发现它存在有限的记忆视野 (例如，梯度消失问题)。尽管提出了各种启发式方法来克服这一点，比如成功的 LSTM 和 GRU 中的门控机制，或者最近的 Fourier 循环单元和 Legendre 记忆单元 (LMU) 中的高阶频率，但对记忆的统一理解仍然是一个挑战。此外，现有方法通常需要对序列长度或时间尺度进行先验假设，并且在此范围之外效果不佳; 这在分布发生转移的情况下可能会带来问题 (例如医疗数据中不同仪器的采样率不同)。最后，其中许多缺乏关于它们捕获长期依赖性质量的理论保证，如梯度有界性。为了设计更好的记忆表示，理想情况下我们应该 (i) 对现有方法有一个统一的认识，(ii) 能够在不对时间尺度做先验假设的情况下处理任何长度的依赖性，以及 (iii) 对其记忆机制有严格的理论理解。

我们的洞见是将记忆作为一个在线函数逼近的技术问题来表述，其中一个函数 $f(t): \mathbb{R}_{+} \rightarrow \mathbb{R}$ 通过存储它在某些基函数上的最优系数来进行总结。这个逼近是相对于一个指定过去每个时间重要性的测度来评估的。给定这个函数逼近公式化，正交多项式 (OPs) 作为一个自然的基 emerge 出来，因为它们的最优系数可以用闭式表示 [14]。正交多项式拥有丰富且经过深入研究的历史 [65]，以及在逼近理论 [68] 和信号处理 [57] 中的广泛应用，它们为这个记忆表示问题带来了一整套技术。我们形式化了一个框架 HiPPO(高阶多项式投影算子)，它产生了算子将任意函数投影到正交多项式空间上，相对于给定的测度。这个通用框架允许我们分析几个测度族，其中这个算子作为一个闭式 ODE 或线性递归关系，允许在输入函数随时间展开时快速增量更新最优多项式逼近。

通过提出支配循环序列模型的形式优化问题，HiPPO 框架 (第 2 节) 概括并解释了之前的方法，开启了适用于不同时间尺度序列数据的新方法，并且带来了一些理论保证。(i) 例如，通过一个简短的推导，我们准确地恢复了 LMU[71] 作为一个特例 (第 2.3 节)，它提出了一个投影到固定长度滑动窗口的更新规则。HiPPO 还揭示了经典技术如 LSTM 和 GRU 中的门控机制的新见解，它们在只使用低阶近似度数的一个极端情况下出现 (第 2.5 节)。(ii) ***通过选择更合适的测度，HiPPO 产生了一种新机制 (Scaled Legendre，或 LegS)，它始终考虑函数的整个历史记录，而不是滑动窗口***。这种灵活性消除了对序列长度的超参数或先验需求，允许 LegS 推广到不同的输入时间尺度。(iii) 与动力系统和逼近理论的联系使我们能够展示 HiPPO-LegS 的一些理论优势: 不变于输入时间尺度、更有效的渐近更新，以及梯度流和逼近误差的边界 (第 3 节)。

我们将 HiPPO 记忆机制集成到 RNN 中，并在用于评测长期依赖性的标准任务上实证地展示了它们优于基线方法的表现。在 permuted MNIST 数据集上，我们无需超参数的 HiPPO-LegS 方法实现了 98.3% 的新的最佳精度记录，超过了之前 RNN 最佳结果 1 个多点，甚至优于具有全局上下文的变压器模型 (第 4.1 节)。接下来，我们在一个新颖的轨迹分类任务上展示了 HiPPO-LegS 的时间尺度稳健性，它能够推广到看不见的时间尺度并处理缺失数据，而 RNN 和神经 ODE 基线则失败了 (第 4.2 节)。最后，我们验证了 HiPPO 的理论，包括计算效率和可扩展性，允许在数百万个时间步长上快速精确地在线重构函数 (第 4.3 节)。重现我们实验的代码可在 https://github.com/HazyResearch/hippo-code 获得。

## 2 HiPPO 框架: 高阶多项式投影算子

### 2.1 HiPPO 问题设置

给定一个输入函数 $f(t) \in \mathbb{R}$ 在 $t \geq 0$ 上，许多问题需要在每个时间 $t \geq 0$ 时对累积历史 $f_{\leq t}:=\left.f(x)\right|_{x \leq t}$ 进行操作，<u>*以理解到目前为止看到的输入并做出未来预测*</u>。由于函数空间非常庞大，历史记录不可能被完美地记住，**必须被压缩**; 我们提出通过将其投影到有限维子空间的一般方法。因此，我们的目标是维护 (在线) 这个历史的压缩表示。为了完整地指定这个问题，我们需要两个组成部分: **==量化逼近的一种方式，以及一个合适的子空间==**。

**相对于一个测度的函数逼近** ：为了评价这种逼近的程度，我们需要在函数空间中定义一个距离。**==任何概率测度 $\mu$ 在 $[0， \infty)$ 上为可平方积分函数空间赋予内积 $\langle f,g\rangle_\mu=\int_0^{\infty} f(x) g(x) \mathrm{d} \mu(x)$，诱导出一个 Hilbert 空间结构 $\mathcal{H}_\mu$ 和相应的范数 $\|f\|_{L_2(\mu)}=\langle f, f\rangle_\mu^{1 / 2}$。==**

**多项式基展开**：**==这个函数空间的任何 N 维子空间 $\mathcal{G}$ 都是近似的合适候选==**。**<u>参数 N 对应于近似的阶数</u>**，或压缩的大小; 投影历史可以由它在 $\mathcal{G}$ 的任何一组基上的 N 个系数表示。在本文的其余部分，**<u>我们使用多项式作为一个自然的基，因此 $\mathcal{G}$ 是次数小于 N 的多项式集合</u>**。我们注意到多项式基是非常通用的; 例如，Fourier 基 $\sin (n x)， \cos (n x)$ 可以被看作是单位圆上的多项式 $(e^{2 \pi i x})^n$(参见附录 D.4)。在附录 C 中，我们还形式化了一个更通用的框架，**<u>*允许通过对测度进行傾斜来使用除多项式之外的其他基。*</u>**

**在线逼近**：**<u>*由于我们关心逼近每个时间 $t$ 的 $f_{<t}$，我们还让测度随时间变化*</u>**。对于每个 $t$，令 $\mu^{(t)}$ 是一个支撑在 $(-\infty, t]$ 上的测度 (因为 $f \leq t$ 只定义到时间 $t$)。**<u>*总的来说，我们寻求 $\mathcal{G}$ 中的某个 $g^{(t)}$ 使 $\|f_{\leq t}-g^{(t)}\|_{L_2(\mu^{(t)})}$ 最小化*</u>**。直观地说，测度 $\mu$ 控制着输入域各部分的重要性，基决定了允许的近似。挑战在于如何在给定 $\mu^{(t)}$ 的情况下以闭式解决优化问题，以及这些系数如何在 $t \rightarrow \infty$​时在线维护。

### 2.2 一般 HiPPO 框架

我们简要概述了解决这个问题背后的主要思路，这为许多测度族 $\mu^{(t)}$ 提供了一种出人意料的简单而通用的策略。**<u>*这个框架建立在信号处理文献中研究了很久的正交多项式及相关变换的丰富历史基础之上*</u>**。我们的形式抽象 (定义 1) 在几个方面区别于之前关于滑动变换的工作，我们将在附录 A.1 中详细讨论。例如，我们的时变测度概念允许更合理地选择 $\mu^{(t)}$，这将导致质量不同的行为的解。附录 C 包含了我们框架的全部细节和形式。

**通过连续动力学计算投影**：如前所述，近似函数可以由它在任何基上展开的 N 个系数表示; 第一个关键步骤是选择 $\mathcal{G}$ 的一个合适基 $\{g_n\}_{n<N}$。利用逼近理论中的经典技术，一个自然的基是测度 $\mu^{(t)}$ 的正交多项式集合，**<u>它形成了该子空间的一个正交基。然后最优基展开的系数就简单地是 $c_n^{(t)}:=\langle f_{\leq t}， g_n\rangle_{\mu^{(t)}}$</u>**。

第二个关键思想是对 $t$ 求导这个投影，**<u>*其中通过对积分 (来自内积 $\langle f_{\leq t}, g_n\rangle_{\mu^{(t)}}$) 求导将常常导致一个自相似关系，使 $\frac{d}{dt} c_n(t)$ 可以由 $(c_k(t))_{k\in[N]}$ 和 $f(t)$ 表示。因此系数 $c(t) \in \mathbb{R}^N$ 应该按照一个 ODE 演化*</u>**，其动力学由 $f(t)$ 决定。  

HiPPO 概要: 在线函数逼近。

定义 1.给定一个时变测度族 $\mu^{(t)}$ 支撑在 $(-\infty, t]$ 上，一个 N 维多项式子空间 $\mathcal{G}$，以及一个连续函数 $f: \mathbb{R}_{\geq 0} \rightarrow \mathbb{R}$，HiPPO 在每个时间 $t$ 定义一个投影算子 $\operatorname{proj}_t$ 和一个系数提取算子 $\operatorname{coef}_t$，具有以下性质:

(1) $\operatorname{proj}_t$ 取函数 $f$ 限制到时间 $t$ 之前的部分 $f_{\leq t}:=\left.f(x)\right|_{x \leq t}$，并将其映射到 $\mathcal{G}$ 中的一个多项式 $g^{(t)}$，使 $\|f_{\leq t}-g^{(t)}\|_{L_2(\mu^{(t)})}$ 最小化。

(2) $\operatorname{coef}_t: \mathcal{G} \rightarrow \mathbb{R}^N$ 将多项式 $g^{(t)}$ 映射到相对于测度 $\mu^{(t)}$ 定义的正交多项式基的系数 $c(t) \in \mathbb{R}^N$。

$\operatorname{coef}\circ\operatorname{proj}$ 的复合称为 hippo，**==它是一个算子将函数 $f: \mathbb{R}_{\geq 0} \rightarrow \mathbb{R}$ 映射到最优投影系数 $c: \mathbb{R}_{\geq 0} \rightarrow \mathbb{R}^N$，即 $(hippo(f))(t)=\operatorname{coef}_t(\operatorname{proj}_t(f))$。==**

对于每个 $t$，最优投影 $\operatorname{proj}_t(f)$ 的问题由上述内积很好地定义，但天然计算是无法解决的。我们的推导 (附录 D) 将展示系数函数 $c(t)=\operatorname{coef}_t(\operatorname{proj}_t(f))$ 的形式为 $\frac{d}{dt} c(t)=A(t)c(t)+B(t)f(t)$，其中 $A(t) \in \mathbb{R}^{N \times N}， B(t) \in \mathbb{R}^{N \times 1}$。因此我们的结果展示了如何通过求解 ODE 或更具体地运行一个离散递推来在线高效地获得 $c^{(t)}$。当离散化后，HiPPO 接收一个实数序列并产生一个 N 维向量序列。  

图 1 说明了在使用均匀测度时的整体框架。接下来，我们给出主要结果，展示了框架在几种具体实例下的 hippo 形式。

![image.png](https://raw.githubusercontent.com/aletolia/Pictures/main/202403301442079.png)

图 1:HiPPO 框架的说明。(1) 对于任何函数 $f$，(2) 在每个时间 $t$，都存在 $f$ 关于一个加权过去的测度 $\mu^{(t)}$ 在多项式空间上的一个最优投影 $g^{(t)}$。(3) 对于适当选择的基，相应的表示 $f$ 历史压缩的系数 $c(t) \in \mathbb{R}^N$ 满足线性动力学。(4) 将这样的线性动态离散化，得到一个高效的闭式递推关系，用于在线压缩时间序列 $(f_k)_{k \in \mathbb{N}}$。

### 2.3 高阶投影: 测度族和 HiPPO ODEs

我们的主要理论结果是 HiPPO 在各种测度族 $\mu^{(t)}$ 下的实例化。我们提供了两个自然的滑动窗口测度及其对应的投影算子的例子。对记忆机制的统一视角使我们能够以同样的策略 (见附录 D.1，D.2) 推导出这些闭式解。第一个以原则性的方式解释了核心的 Legendre 记忆单元 (LMU)[71] 更新，并描述了其局限性，而另一个是新颖的，展示了 HiPPO 框架的通用性。附录 D 对比了这些测度的权衡 (图 5)，包含了它们推导的证明，并推导了其他基 (如 Fourier，从而恢复 Fourier 循环单元 [79]) 和 Chebyshev) 的额外 HiPPO 公式。

**==平移 Legendre(LegT) 测度对最近历史 $[t-\theta, t]$ 赋予均匀权重==**。有一个超参数 $\theta$ 表示滑动窗口的长度，或被总结的历史长度。而平移 Laguerre(LagT) 测度则使用**指数衰减测度，赋予更近期历史更高的重要性**。

$$
\text{LegT: } \mu^{(t)}(x)=\frac{1}{\theta} \mathbb{I}_{[t-\theta, t]}(x) \quad \text { LagT: } \mu^{(t)}(x)=e^{-(t-x)} \mathbb{I}_{(-\infty, t]}(x)= \begin{cases}e^{x-t} & \text { if } x \leq t \\ 0 & \text { if } x>t\end{cases}
$$

定理 1.对于 LegT 和 LagT，满足定义 1 的 hippo 算子由线性时不变 (LTI) ODE $\frac{d}{dt}c(t)=-Ac(t)+Bf(t)$ 给出，其中 $A \in \mathbb{R}^{N \times N}， B \in \mathbb{R}^{N \times 1}$:

LegT:

$$
A_{nk}=\frac{1}{\theta}\left\{\begin{array}{ll}
(-1)^{n-k}(2n+1) & \text { if } n \geq k \\
2n+1 & \text { if } n \leq k
\end{array}, \quad B_n=\frac{1}{\theta}(2n+1)(-1)^n\right.
$$

LagT:

$$
A_{nk}=\left\{\begin{array}{ll}
1 & \text { if } n \geq k \\
0 & \text { if } n<k
\end{array}, \quad B_n=1\right.
$$

方程 (1) 证明了 LMU 更新 [71，方程 (1)]。另外，我们的推导 (附录 D.1) 表明除了投影之外，还有另一个逼近源。这个滑动窗口更新规则需要访问 $f(t-\theta)$，而它已不再可用; 它假设当前系数 $c(t)$ 是函数 $f(x)_{x \leq t}$ 的足够精确模型，以至于 $f(t-\theta)$ 可以从中恢复。

### 2.4 HiPPO 递推关系: 从连续时间到离散时间通过 ODE 离散化  

由于实际数据本质上是离散的 (如序列和时间序列)，我们讨论如何使用标准技术对 HiPPO 投影算子进行离散化，以使连续时间 HiPPO ODE 成为离散时间线性递推关系。

在连续情况下，这些算子使用输入函数 $f(t)$ 并产生输出函数 $c(t)$。离散时间情况 (i) 消费输入序列 $(f_k)_{k \in \mathbb{N}}$，(ii) 隐式定义一个函数 $f(t)$，其中 $f(k \cdot \Delta t)=f_k$，使用某个步长 $\Delta t$，(iii) 通过 ODE 动力学产生函数 $c(t)$，(iv) 离散回到输出序列 $c_k:=c(k \cdot \Delta t)$。

离散化 ODE $\frac{d}{dt}c(t)=u(t,c(t),f(t))$ 的基本方法选择一个步长 $\Delta t$，并执行离散更新 $c(t+\Delta t)=c(t)+\Delta t \cdot u(t,c(t),f(t))$。一般来说，这个过程对离散化步长超参数 $\Delta t$ 是敏感的。

最后，我们注意到这提供了一种无缝处理带时间戳数据的方式，即使有缺失值: 时间戳之间的差异表示在离散化中使用的 (自适应)$\Delta t$[13]。附录 B.3 包含了关于离散化的完整讨论。 

### 2.5 低阶投影: 门控 RNN 的记忆机制  

作为一个特例，我们考虑如果不在投影问题中纳入高阶多项式会发生什么。具体地，如果 $N=1$，那么 HiPPO-LagT(2) 的离散版本就变成 $c(t+\Delta t)=c(t)+\Delta t(-Ac(t)+Bf(t))=(1-\Delta t)c(t)+\Delta tf(t)$，因为 $A=B=1$。**<u>如果输入 $f(t)$ 可以依赖隐藏状态 $c(t)$，并且离散步长 $\Delta t$ 被自适应地选择 (作为输入 $f(t)$ 和状态 $c(t)$​的函数)，如同 RNN 一样，那么这就精确成为一个门控 RNN</u>**。例如，通过并行堆叠多个单元并选择特定的更新函数，我们可以将 GRU 更新单元作为一个特例获得。与使用一个隐藏特征并将其投影到高阶多项式的 HiPPO 不同，这些模型使用许多隐藏特征但只将它们投影到次数 1。这种观点通过展示如何从第一原理推导出这些经典技术，揭示了对它们的新见解。

## 3 HiPPO-LegS: 经过缩放的测度以获得时间尺度稳健性

揭示了在线函数逼近与记忆之间的紧密联系，使我们能够通过适当选择测度来产生具有更好理论性质的记忆机制。尽管滑动窗口在信号处理中很常见 (附录 A.1)，但对于记忆更直观的方法应随时间缩放窗口以避免遗忘。

我们新颖的经过缩放的 Legendre 测度 (LegS) 对所有历史 $[0, t]$ 赋予均匀权重:$\mu^{(t)}=\frac{1}{t} \mathbb{I}_{[0， t]}$。附录 D 图 5 以可视化的方式比较了 LegS、LegT 和 LagT，显示了缩放测度的优势。

简单地指定期望的测度，专门化 HiPPO 框架 (第 2.2 和 2.4 节) 就产生了一种新的记忆机制 (证明见附录 D.3)。  

定理 2. HiPPO-LegS 的连续时间 (3) 和离散时间 (4) 动力学为:

$$
\begin{aligned}
\frac{d}{dt}c(t) & =-\frac{1}{t}Ac(t)+\frac{1}{t}Bf(t)\\
c_{k+1} & =\left(1-\frac{A}{k}\right)c_k+\frac{1}{k}Bf_k
\end{aligned}
$$

$$
A_{nk}=\left\{\begin{array}{ll}
(2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n>k\\
n+1 & \text{if } n=k,\\
0 & \text{if } n<k
\end{array}\quad B_n=(2n+1)^{\frac{1}{2}}\right.
$$

我们证明 HiPPO-LegS 享有有利的理论性质: 它对输入时间尺度是不变的，计算快捷，并且梯度和逼近误差有界。所有证明见附录 E。

时间尺度稳健性。由于 LegS 的窗口大小是自适应的，投影到这个测度上在直觉上对时间尺度是稳健的。**<u>形式上，HiPPO-LegS 算子是时间尺度等变的: 扭曲输入 $f$ 不会改变逼近系数。</u>**

命题 3.对任何标量 $\alpha>0$，如果 $h(t)=f(\alpha t)$，那么 $hippo(h)(t)=hippo(f)(\alpha t)$。换言之，如果 $\gamma: t \mapsto \alpha t$ 是任何扭曲函数，那么 $hippo(f \circ \gamma)=hippo(f) \circ \gamma$。

非正式地说，HiPPO-LegS 没有时间尺度超参数反映了这一点; 特别是离散递归 (4) 对离散化步长是不变的。相比之下，LegT 有一个窗口大小超参数 $\theta$，LegT 和 LagT 在离散时间情况下都有一个步长超参数 $\Delta t$。这个超参数在实践中很重要; 第 2.5 节展示了 $\Delta t$ 与 RNN 的门控机制有关，已知后者对其参数化是敏感的 [31，39，66]。我们在第 4.2 节中实证了时间尺度稳健性的好处。

计算效率。为了计算 HiPPO 更新的单步，主要操作是乘以 (离散化的) 方阵 $A$。更一般的离散化专门需要对任何形式 $I+\Delta t \cdot A$ 和 $(I-\Delta t \cdot A)^{-1}$ 的矩阵进行快速乘法，其中 $\Delta t$ 为任意步长。尽管这在一般情况下是 $O(N^2)$ 操作，但 LegS 算子使用具有特殊结构的固定 $A$ 矩阵，事实证明对任何离散化都有快速乘法算法。

命题 4.在任何广义双线性变换离散化 (参见附录 B.3) 下，方程 (4) 中 HiPPO-LegS 递推的每一步可以在 $O(N)$ 操作下计算。

第 4.3 节验证了 HiPPO 层在实践中的效率，其中展开定理 2 的离散版本比标准 RNN 中的标准矩阵乘法快 10 倍。

梯度流。为了缓解 RNN 中的梯度消失问题 [56]，人们做了大量工作，其中基于反向传播的学习受到了梯度幅值随时间指数衰减的阻碍。由于 LegS 是为记忆而设计的，它避免了梯度消失问题。

命题 5.对于任何时间 $t_0<t_1$，HiPPO-LegS 算子在时间 $t_1$ 的输出相对于时间 $t_0$ 的输入的梯度范数为 $\left\|\frac{\partial c(t_1)}{\partial f(t_0)}\right\|=\Theta(1/t_1)$。

逼近误差界。LegS 的误差率随输入的平滑性降低。  

命题 6.设 $f: \mathbb{R}_+ \rightarrow \mathbb{R}$ 为可微函数，在时间 $t$ 由最高多项式次数为 $N-1$ 的 HiPPO-LegS 投影为 $g^{(t)}=proj_t(f)$。如果 $f$ 是 L-Lipschitz 的，那么 $\|f_{\leq t}-g^{(t)}\|=O(tL/\sqrt{N})$。如果 $f$ 具有有界的 k 阶导数，那么 $\|f_{\leq t}-g^{(t)}\|=O(t^kN^{-k+1/2})$。

## 相关工作

### A.1 信号处理和正交多项式

#### A.1.1 滑动变换

本文的技术贡献建立在信号处理中逼近理论的丰富历史之上。我们的主要框架——将函数正交化相对于时变测度 (第 2 节)——与经典信号处理变换的 " 在线 " 版本有关。简而言之,这些方法在离散序列的滑动窗口上计算特定的变换。具体地,给定信号 $(f_k)$,它们计算 $c_{n,k}=\sum_{i=0}^{N-1} f_{k+i} \psi(i, n)$,其中 $\{\psi(i, n)\}$ 是一个离散正交变换。我们的技术问题在几个关键方面有所不同:

**特定的离散变换**：文献中考虑的滑动变换的例子包括滑动 DFT[26, 28, 36, 37]、滑动 DCT[43]、滑动离散 (Walsh-)Hadamard 变换 [54, 55, 75]、Haar[51]、滑动离散 Hartley 变换 [44] 和滑动离散 Chebyshev 矩 [12]。虽然每一个都解决了一个特定的变换,但我们提出了一种通用方法 (第 2 节),一次解决了几种变换。此外,我们没有发现针对我们在这里考虑的正交多项式 (特别是 Legendre 和 Laguerre 多项式) 的滑动变换算法。我们在附录 D 中的推导涵盖了 Legendre、(广义)Laguerre、Fourier 和 Chebyshev 连续滑动变换。

**固定长度滑动窗口**：所有提及的工作都在滑动窗口设置中进行,即考虑离散信号上的固定大小上下文窗口。我们基于测度的逼近抽象允许考虑一种新型的经过缩放的测度,其中窗口大小随时间增加,导致了理论 (第 3 节) 和实证 (第 4.2 节) 性质质量不同的方法。我们没有发现任何先前解决这种缩放设置的工作。

**离散与连续时间**：即使在固定长度滑动窗口的情况下,我们对 " 平移测度 " 问题 (例如 HiPPO-LegT,附录 D.1) 的解也是在潜在的连续信号上解决了一个连续时间滑动窗口问题,然后离散化。

另一方面,滑动变换问题直接在离散流上计算变换。**<u>离散变换等价于通过高斯求积在测度上 (方程 (18)) 计算投影系数,它假设离散输入是从求积节点处的信号中子采样得到的 [14]</u>**。然而,由于这些节点在一般情况下是不均匀间隔的,因此滑动离散变换与对潜在连续信号的离散化是不一致的。

因此,我们的主要抽象 (定义 1) 与标准变换有着根本不同的解释,我们首先计算潜在的连续时间问题 (如方程 (20)) 的动力学的方法也因此是新颖的。我们注意到,使用基于标准离散时间的方法来处理我们新颖的缩放测度是 fundamentally 有困难的。这些离散滑动方法需要固定大小的上下文,以便具有一致的变换尺寸,而缩放测度则需要随时间解决输入点数量增加的变换。

#### A.1.2 ML 中的正交多项式

更广泛地说,正交多项式和正交多项式变换最近在机器学习的各个方面都有应用。例如,Dao 等人 [19] 利用正交多项式与求积之间的联系,推导出计算机器学习中核函数特征的规则。更直接地,[67] 将受正交多项式变换 ([22]) 直接启发的参数化结构矩阵族作为神经网络的层进行应用。某些特定的正交多项式族如 Chebyshev 多项式在数值分析和优化中有许多已知的经典用途,因为它们具有理想的逼近性质。**<u>最近,它们被应用到图卷积神经网络 [24] 等 ML 模型中,且其推广形式如 Gegenbauer 和 Jacobi 多项式已被用于分析优化动力学</u>** [7, 76]。表示为蝴蝶矩阵乘积的正交多项式和 Fourier 变换的推广已经在自动算法设计 [20]、模型压缩 [1] 以及替代语音识别中的手工预处理 [21] 中找到了应用。正交多项式已知具有各种效率结果 [22],我们猜测 HiPPO 方法的效率性质 (命题 4) 可以推广到除本文考虑之外的任意测度。

### A.2 机器学习中的记忆

**序列模型中的记忆**：在语言、强化学习和持续学习等领域,序列或时间数据可能涉及越来越长的依赖关系。然而,直接的参数化建模无法处理长度未知和可能无界的输入。许多现代解决方案,如注意力 [70] 和扩张卷积 [5],都是在有限窗口上的函数,因此避免了对显式记忆表示的需求。虽然这对于某些任务是足够的,但这些方法只能处理有限的上下文窗口,而不是整个序列。直接增加窗口长度会带来重大的计算和内存挑战。这促生了各种在计算和存储受限的情况下扩展这个固定上下文窗口的方法 [6,15,18,42,59,60,64,74]。

我们转而关注连续和离散信号的在线处理和记忆化这一核心问题,并期望研究这个基础性问题将有助于改进各种模型。

**循环记忆**：循环神经网络是在线对序列数据建模的自然工具,具有无界上下文的吸引力,换言之,它们可以无限总结历史。然而,由于优化过程中的困难 (梯度消失/爆炸 [56]),必须特别注意赋予它们更长的记忆。无处不在的 LSTM[34] 和 GRU[17] 等简化形式使用门控制更新以平滑优化过程。通过更精心的参数化,单单增加门控就使 RNN 显著更加稳健并能够解决长期依赖问题 [31]。Tallec 和 Ollivier[66] 表明,门控实际上对循环动力学是基本的,因为它们允许时间扩张。赋予 RNN 更好记忆的其他方法还包括注入噪声 [32] 或非饱和门 [9],但可能存在不稳定性问题。有一系列工作通过 (近似) 正交矩阵来控制循环更新的谱以控制梯度 [3],但被发现在不同任务中的稳健性较差 [33]。

### A.3 直接相关的方法

LMU Legendre 记忆单元 [71,72,73] 的主要结果是使用 LegT 测度 (第 2.3 节) 对我们框架的一个直接实例。最初的 LMU 受神经生物学进展的启发,从与我们相反的方向来解决这个问题: 它考虑在频域中逼近脉冲神经元,而我们直接在时域中解决一个可解释的优化问题。更具体地说,他们考虑了时滞线性时不变 (LTI) 动力系统,并用 Padé近似展开来逼近动力学;Voelker 等人 [71] 观察到结果也可以用 Legendre 多项式来解释,但没有意识到它是一个自然投影问题的最优解。这种方法涉及更重的机器,我们无法找到对更新机制的完整证明 [71,72,73]。

相比之下,我们的方法直接提出了相关的在线信号逼近问题,这与正交多项式族有关,并导出了几种相关记忆机制的简单推导 (附录 D)。我们在时域而不是频域的解释,以及对 LegT 测度的相关推导 (附录 D.1),揭示了源于滑动窗口的一组不同的逼近,这在实证上得到了确认 (附录 F.8)。

由于我们工作的动机与 Voelker 等人 [71] 有实质不同,但在特殊情况下找到了相同的记忆机制,我们强调将这些序列模型与生物神经系统之间的潜在联系作为未来工作的一个探索方向,例如在频域中对我们方法的替代解释。

我们注意到,LMU 一词实际上是指一种特定的循环神经网络架构,它将投影算子与其他特定的神经网络组件交织在一起。相比之下,我们使用 HiPPO 来指代单独的投影算子 (定理 1),它是一个与模型无关的函数到函数或序列到序列的算子。HiPPO 被集成到第 4 节的 RNN 架构中,对 LMU 架构做了轻微改进,如附录 F.2 和 F.3 中的消融实验所示。作为一个独立的模块,HiPPO 可以用作其他类型模型中的一个层。

#### 傅立叶循环单元

傅立叶循环单元（FRU）\[79\] 使用傅立叶基（余弦和正弦）来表示输入信号，其灵感来自离散傅立叶变换。具体来说，每个循环单元计算输入信号关于随机选择的频率的离散傅立叶变换。目前尚不清楚如何利用关于其他基（例如 Legendre、Laguerre、Chebyshev）的离散变换来产生类似的记忆机制。我们证明了 FRU 也是 HiPPO 框架的一种实例（附录 D.4），其中傅立叶基可以被视为单位圆上的正交多项式 $z^n$。Zhang 等人\[79\] 证明了，如果选择合适的时间尺度超参数，FRU 具有有界梯度，从而避免了梯度消失和梯度爆炸。这主要是因为如果选择了离散化步长 $\Delta t=\Theta\left(\frac{1}{T}\right)$，并且时间跨度 $T$ 已知，则 $(1-\Delta t)^T=\Theta(1)$（见附录 B. 3 和 E）。很容易证明，这个属性不是 FRU 固有的，而是滑动窗口方法的属性，并且我们所有转换的 HiPPO 方法都共享这个属性（附录 (D) 除外）。我们展示了更强的属性，即 HiPPO-LegS，它使用缩放而不是滑动窗口，也享有有界梯度保证，而不需要一个良好指定的时间尺度超参数（命题 5）。

#### 神经 ODEs

**<u>HiPPO 产生描述系数动态的线性 ODEs</u>**。最近的工作还将 ODEs 纳入了机器学习模型中。Chen 等人\[13\] 引入了神经 ODEs，利用神经网络参数化的一般非线性 ODEs 在正规化流和时间序列建模的背景下。神经 ODEs 在建模不规则采样时间序列方面表现出有希望的结果\[40\]，尤其是与 RNNs\[61\] 结合使用时。尽管神经 ODEs 很具表现力\[27, 78\]，由于其复杂的参数化，它们通常在训练过程中速度较慢\[29, 53, 58\]，因为它们需要更复杂的 ODE 求解器。**<u>另一方面，HiPPO ODEs 是线性的，并且可以利用线性系统中的经典离散化技术（如欧拉方法、双线性方法和零阶保持（ZOH））快速求解\[35\]。</u>**

### B 技术准备工作

我们在这里收集了一些技术背景，将用于介绍通用的 HiPPO 框架并推导特定的 HiPPO 更新规则。

#### B.1 正交多项式

正交多项式是处理函数空间的标准工具\[14,65\]。**<u>==每个测度 $\mu$ 引导出一个唯一的（至标量倍数）正交多项式（OPs）序列 $P_0(x), P_1(x), \ldots$，满足 $\operatorname{deg}\left(P_i\right)=i$ 和 $\left\langle P_i, P_j\right\rangle_\mu:=\int P_i(x) P_j(x) \mathrm{d} \mu(x)=0$==</u>**，对于所有 $i \neq j$。这是通过对于 $\langle\cdot,\rangle_\mu$ 进行格拉姆 - 施密特正交化得到的多项式基 $\left\{x^i\right\}$ 的序列。OPs 形成正交基的事实很有用，因为用来近似函数 $f$ 的最佳正交多项式 $g(\deg(g)<N)$ 可以通过如下方式给出：
$$
\sum_{i=0}^{N-1} c_i P_i(x) /\left\|P_i\right\|_\mu^2 \quad \text { 其中 } c_i=\left\langle f, P_i\right\rangle_\mu=\int f(x) P_i(x) \mathrm{d} \mu(x) .
$$

经典的 OPs 族包括 Jacobi（包括 Legendre 和 Chebyshev 多项式作为特例）、Laguerre 和 Hermite 多项式。傅立叶基也可以解释为复平面中单位圆上的 OPs。

> 在数学中，测度是一种将几何空间的测度 (长度、面积、体积) 和其他常见概念 (如大小、质量和事件的概率) 广义化后产生的概念。传统的黎曼积分是在区间上进行的，为了把积分推广到更一般的集合上，人们就发展出测度的概念。一个特别重要的例子是勒贝格测度，它从 $n$ 维欧式空间 $\mathbb{R}^n$ 出发，概括了传统长度、面积和体积等等的概念。
>
> 研究测度的学问被统称为测度论，因为指定的数值通常是非负实数，所以测度论通常会被视为实分析的一个分支，它在数学分析和概率论有重要的地位。
>
> **正式定义** 
>
> 定义 $-(X, \Sigma)$ 为可测空间，函数 $\mu: \Sigma \rightarrow[0, \infty)$ 若满足:
>
> - $\mu(\varnothing)=0$ (空集合的测度为零)
> - 可数可加性 $\left(\sigma\right.$-可加性)：若集合序列 $\left\{E_n \in \Sigma\right\}_{n \in \mathbb{N}}$ 对所有不相等正整数 $i \neq j$ 都有 $E_i \cap E_j=\varnothing$ ，则

> $$
> \mu\left(\bigcup_{n \in \mathbb{N}} E_n\right)=\sum_{n=1}^{\infty} \mu\left(E_n\right) 。
> $$
>
> 那 $\mu$ 被称为定义在 $\Sigma$ 上的一个非负测度，或简称为测度。为了叙述简便起见，也可称 $(X, \Sigma, \mu)$​ 为一测度空间。
>
> 直观上，测度是 "体积” 的推广；因为空集合的 “体积" 当然为零，而且互相独立的一群 (可数个) 物体，总 “体积" 当然要是所有物体 “体积 ${ }^n$ 直接加总 (的极限) 。而要定义 “体积"，必须先要决定怎样的一群子集合，是 “可以测量的"，详细请见 $\sigma$-代数。
>
> 如果将 $\mu$ 的值域扩展到复数，也就是说 $\mu: \Sigma \rightarrow \mathbb{C}$ ，那 $\mu$ 会被进一步称为复数测度。 ${ }^{\text {[1] }}$
>
> 定义的分歧
>
> 若照着上述定义，根据可数可加性，不少母集合本身的测度值会变成无穷大 (如对 $\mathbb{R}^n$ 本身取勒贝格测度)，所以实际上不存在。但某些书籍 ${ }^{[2]}$​​ 会形式上将无穷大视为一个数，而容许测度取值为无穷大；这样定义的书籍，会把只容许有限实数值的测度称为 (非负) 有限测度。但这样"定义"，会造成可数可加性与数列收敛的定义产生矛盾。
>
> 所以要延续体积是一种 " 测度 " 的这种直观概念（也就是严谨的定义勒贝格测度），那就必须把 $\sigma$-代数换成条件比较宽松的半集合环，然后以此为基础去定义一个对应到 " 体积" 的前测度。
>
> 更进一步的，如果对测度空间 $(X, \Sigma, \mu)$ 来说，母集合 $X$ 可表示为 $\Sigma$ 内的某可测集合序列 $\left\{E_n \in \Sigma\right\}_{n \in \mathbb{N}}$​ 的并集：
> $$
> X=\bigcup_{n \in \mathbb{N}} E_n
> $$
>
> 且 $\mu$ 只容许取有限值，则 $\mu$ 会被进一步的称为 (非负) $\sigma$​-有限测度。
>
> 单调性 
>
> 测度 $\mu$ 的单调性：若 $E_1$ 和 $E_2$ 为可测集，而且 $E_1 \subseteq E_2$ ，则 $\mu\left(E_1\right) \leq \mu\left(E_2\right)$ 。
>
> **可数个可测集的并集的测度**
>
> 若 $E_1, E_2, E_3 \cdots$ 为可测集 (不必是两两不交的)，则集合 $E_n$​ 的并集是可测的，且有如下不等式（“次可列可加性"）：

> $$
> \mu\left(\bigcup_{i=1}^{\infty} E_i\right) \leq \sum_{i=1}^{\infty} \mu\left(E_i\right)
> $$
>
> 如果还满足并且对于所有的 $n ， E_n \subseteq E_{n+1}$ ，则如下极限式成立:

> $$
> \mu\left(\bigcup_{i=1}^{\infty} E_i\right)=\lim _{i \rightarrow \infty} \mu\left(E_i\right) .
> $$
>
> **可数个可测集的交集的测度** 
>
> 若 $E_1, E_2, \cdots$ 为可测集，并且对于所有的 $n ， E_{n+1} \subseteq E_n$ ，则 $E_n$ 的交集是可测的。进一步说，如果至少一个 $E_n$​ 的测度有限，则有极限：

> $$
> \mu\left(\bigcap_{i=1}^{\infty} E_i\right)=\lim _{i \rightarrow \infty} \mu\left(E_i\right)
> $$
>
> 如若不假设至少一个 $E_n$ 的测度有限，则上述性质一般不成立。例如对于每一个 $n \in \mathbb{N}$ ，令

> $$
> E_n=[n, \infty) \subseteq \mathbb{R}
> $$
>
> 这里，全部集合都具有无限测度，但它们的交集是空集。

##### B.1.1 Legendre 多项式的性质

Legendre 多项式 根据标准定义的规范 Legendre 多项式 $P_n$，它们相对于测度 $\omega^{\text {leg }}=1_{[-1,1]}$ 是正交的：

$$
\frac{2 n+1}{2} \int_{-1}^1 P_n(x) P_m(x) \mathrm{d} x=\delta_{n m}\tag{5}
$$

同样，它们满足

$$
\begin{aligned}
P_n(1) & =1 \\
P_n(-1) & =(-1)^n .
\end{aligned}
$$

移位和缩放的 Legendre 多项式 我们还将考虑将 Legendre 多项式缩放到在区间 $[0, t]$ 上是正交的。对（5）进行变量替换得到

$$
\begin{aligned}
(2 n+1) \int_0^t P_n\left(\frac{2 x}{t}-1\right) P_m\left(\frac{2 x}{t}-1\right) \frac{1}{t} \mathrm{~dx} & =(2 n+1) \int P_n\left(\frac{2 x}{t}-1\right) P_m\left(\frac{2 x}{t}-1\right) \omega^{\operatorname{leg}}\left(\frac{2 x}{t}-1\right) \frac{1}{t} \mathrm{~d} x \\
& =\frac{2 n+1}{2} \int P_n(x) P_m(x) \omega^{\operatorname{leg}}(x) \mathrm{d} x \\
& =\delta_{n m} .
\end{aligned}
$$

因此，相对于测度 $\omega_t=1_{[0, t]} / t$（对于所有 $t$，这是一个概率测度），归一化的正交多项式是

$$
(2 n+1)^{1 / 2} P_n\left(\frac{2 x}{t}-1\right) .
$$

类似地，基

$$
(2 n+1)^{1 / 2} P_n\left(2 \frac{x-t}{\theta}+1\right)
$$

对于均匀测度 $\frac{1}{\theta} \mathbb{I}_{[t-\theta, t]}$ 也是正交的。

一般来说，对于任何均匀测度，正交基包括 $(2 n+1)^{\frac{1}{2}}$ 乘以相应的线性平移版本的 $P_n$​。

Legendre 多项式的导数 我们注意到 Legendre 多项式的以下递推关系（[2, Chapter 12]）：

$$
\begin{aligned}
(2 n+1) P_n & =P_{n+1}^{\prime}-P_{n-1}^{\prime} \\
P_{n+1}^{\prime} & =(n+1) P_n+x P_n^{\prime}
\end{aligned}
$$

第一个方程得到

$$
P_{n+1}^{\prime}=(2 n+1) P_n+(2 n-3) P_{n-2}+\ldots,\tag{6}
$$

其中求和停在 $P_0$ 或 $P_1$​。

这些方程直接意味着

$$
P_n^{\prime}=(2 n-1) P_{n-1}+(2 n-5) P_{n-3}+\ldots\tag{7}
$$

和

$$
\begin{aligned}
(x+1) P_n^{\prime}(x) & =P_{n+1}^{\prime}+P_n^{\prime}-(n+1) P_n \\
& =n P_n+(2 n-1) P_{n-1}+(2 n-3) P_{n-2}+\ldots .
\end{aligned}\tag{8}
$$

这些将在 HiPPO-LegT 和 HiPPO-LegS 更新的推导中使用。

##### B.1.2 Laguerre 多项式的性质

标准的 Laguerre 多项式 $L_n(x)$ 被定义为相对于权函数 $e^{-x}$ 在 $[0, \infty$ ) 上是正交的，而广义 Laguerre 多项式（也称为关联 Laguerre 多项式）$L_n^{(\alpha)}$ 被定义为相对于权函数 $x^\alpha e^{-x}$ 在 $[0, \infty)$ 上是正交的：

$$
\int_0^{\infty} x^\alpha e^{-x} L_n^{(\alpha)}(x) L_m^{(\alpha)}(x) \mathrm{d} x=\frac{(n+\alpha) !}{n !} \delta_{n, m} \tag{9}
$$

此外，它们满足

$$
L_n^{(\alpha)}(0)=\left(\begin{array}{c}
n+\alpha \\
n
\end{array}\right)=\frac{\Gamma(n+\alpha+1)}{\Gamma(n+1) \Gamma(\alpha+1)} .\tag{10}
$$

标准的 Laguerre 多项式对应于广义 Laguerre 多项式的 $\alpha=0$​ 的情况。

广义 Laguerre 多项式的导数 我们注意到广义 Laguerre 多项式的以下递推关系（[2, Chapter 13.2]）：

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} x} L_n^{(\alpha)}(x) & =-L_{n-1}^{(\alpha+1)}(x) \\
L_n^{(\alpha+1)}(x) & =\sum_{i=0}^n L_i^{(\alpha)}(x) .
\end{aligned}
$$

这些方程意味着

$$
\frac{d}{d t} L_n^{(\alpha)}(x)=-L_0^{(\alpha)}(x)-L_1^{(\alpha)}(x)-\cdots-L_{n-1}^{(\alpha)}(x) .
$$

##### B.1.3 Chebyshev 多项式的性质

设 $T_n$ 为经典的 Chebyshev 多项式（第一类），定义为相对于权函数 $\left(1-x^2\right)^{1 / 2}$ 在 $[-1,1]$ 上是正交的，并且 $p_n$ 是 $T_n$ 的归一化版本（即，其范数为 1）：

$$
\begin{aligned}
& \omega^{\text {cheb }}=\left(1-x^2\right)^{-1 / 2} \mathbb{I}_{(-1,1)}, \\
& p_n(x)=\sqrt{\frac{2}{\pi}} T_n(x) \quad \text { 对于 } n \geq 1, \\
& p_0(x)=\frac{1}{\sqrt{\pi}} .
\end{aligned}
$$

注意 $\omega^{\text {cheb }}$ 并非归一化（其积分为 $\pi$）。

**Chebyshev 多项式的导数** Chebyshev 多项式满足

$$
2 T_n(x)=\frac{1}{n+1} \frac{d}{d x} T_{n+1}(x)-\frac{1}{n-1} \frac{d}{d x} T_{n-1}(x) \quad n=2,3, \ldots
$$

通过将这个级数进行折叠，我们得到

$$
\frac{1}{n} T_n^{\prime}=\left\{\begin{array}{ll}
2\left(T_{n-1}+T_{n-3}+\cdots+T_2\right)+T_0 & n \text { 是奇数 } \\
2\left(T_{n-1}+T_{n-3}+\cdots+T_1\right) & n \text { 是偶数 }
\end{array} .\right.\tag{11}
$$

**Translated Chebyshev 多项式** 我们还将考虑将 Chebyshev 多项式进行平移和缩放，==**使其在固定长度 $\theta$ 的区间 $[t-\theta, t]$ 上是正交的。**==
==**归一化（概率）测度为==**

$$
\omega(t, x)=\frac{2}{\theta \pi} \omega^{\text {cheb }}\left(\frac{2(x-t)}{\theta}+1\right)=\frac{1}{\theta \pi}\left(\frac{x-t}{\theta}+1\right)^{-1 / 2}\left(-\frac{x-t}{\theta}\right)^{-1 / 2} \mathbb{I}_{(t-\theta, t)} .
$$

正交多项式基为

$$
p_n(t, x)=\sqrt{\pi} p_n\left(\frac{2(x-t)}{\theta}+1\right) .
$$

用原始的 Chebyshev 多项式表示，这些是

$$
\begin{aligned}
p_n(t, x) & =\sqrt{2} T_n\left(\frac{2(x-t)}{\theta}+1\right) \quad \text { 对于 } n \geq 1, \\
p_0^{(t)} & =T_0\left(\frac{2(x-t)}{\theta}+1\right) .
\end{aligned}
$$

#### B.2 莱布尼茨积分法则

作为推导 HiPPO 更新规则的标准策略的一部分（附录 C），我们将会对具有变化限制的积分进行微分。例如，当分析缩放的 Legendre（LegS）测度时，我们可能希望对表达式 $\int f(t, x) \mu(t, x) \mathrm{d} x=\int_0^t f(t, x) \frac{1}{t} \mathrm{~d} x$ 关于 $t$ 进行微分。

通过这样的积分进行微分可以通过莱布尼茨积分法则进行形式化，其基本版本如下：

$$
\frac{\partial}{\partial t} \int_{\alpha(t)}^{\beta(t)} f(x, t) \mathrm{d} x=\int_{\alpha(t)}^{\beta(t)} \frac{\partial}{\partial t} f(x, t) \mathrm{d} x-\alpha^{\prime}(t) f(\alpha(t), t)+\beta^{\prime}(t) f(\beta(t), t) .
$$

在我们的推导中（附录 $[\mathrm{D})$），我们省略了正式的方法，而是使用以下技巧。我们用指示函数替换被积函数的限制；并在微分时使用狄拉克 δ 函数 $\delta$（即，使用分布导数的形式）。例如，可以使用此技巧简洁地推导上述公式：

$$
\begin{aligned}
\frac{\partial}{\partial t} \int_{\alpha(t)}^{\beta(t)} f(x, t) \mathrm{d} x & =\frac{\partial}{\partial t} \int f(x, t) \mathbb{I}_{[\alpha(t), \beta(t)]}(x) \mathrm{d} x \\
& =\int \frac{\partial}{\partial t} f(x, t) \mathbb{I}_{[\alpha(t), \beta(t)]}(x) \mathrm{d} x+\int f(x, t) \frac{\partial}{\partial t} \mathbb{I}_{[\alpha(t), \beta(t)]}(x) \mathrm{d} x \\
& =\int \frac{\partial}{\partial t} f(x, t) \mathbb{I}_{[\alpha(t), \beta(t)]}(x) \mathrm{d} x+\int f(x, t)\left(\beta^{\prime}(t) \delta_{\beta(t)}(x)-\alpha^{\prime}(t) \delta_{\alpha(t)}\right)(x) \mathrm{d} x \\
& =\int_{\alpha(t)}^{\beta(t)} \frac{\partial}{\partial t} f(x, t) \mathrm{d} x-\alpha^{\prime}(t) f(\alpha(t), t)+\beta^{\prime}(t) f(\beta(t), t)
\end{aligned}
$$

#### B.3 ODE 离散化

在我们的框架中，时间序列输入将以连续函数的形式进行建模，然后进行离散化。这里我们提供一些有关 ODE 离散化方法的背景知识，包括一种新的离散化方法，适用于我们的新方法遇到的特定类型的 ODE。

ODE 的一般形式为 $\frac{d}{d t} c(t)=f(t, c(t))$。我们还将关注形式为 $\frac{d}{d t} c(t)=A c(t)+B f(t)$ 的线性时不变 ODE，作为一种特殊情况。对 ODE 进行离散化的一般方法，对于步长 $\Delta t$​，是将 ODE 重新写为

$$
c(t+\Delta t)-c(t)=\int_t^{t+\Delta t} f(s, c(s)) \mathrm{d} s，\tag{12}
$$

然后近似 RHS 积分。许多 ODE 离散化方法对应于近似 RHS 积分的不同方式：

欧拉法（也称为前向欧拉）。为了近似方程（12）的 RHS，保留左端点 $\Delta t f(t, c(t))$。对于线性 ODE，我们得到：

$$
c(t+\Delta t)=(I+\Delta t A) c(t)+\Delta t B f(t)。
$$

反向欧拉。为了近似方程（12）的 RHS，保留右端点 $\Delta t f(t+\Delta t, c(t+\Delta t))$。对于线性 ODE，我们得到线性方程和更新：

$$
\begin{aligned}
c(t+\Delta t)-\Delta t A c(t+\Delta t) & =c(t)+\Delta t B f(t) \\
c(t+\Delta t) & =(I-\Delta t A)^{-1} c(t)+\Delta t(I-\Delta t A)^{-1} B f(t)。
\end{aligned}
$$

双线性（又称梯形法则、Tustin 方法）。为了近似方程（12）的 RHS，对端点取平均 $\Delta t \frac{f(t, c(t))+f(t+\Delta t, c(t+\Delta t))}{2}$。对于线性 ODE，我们再次得到线性方程和更新：

$$
\begin{aligned}
c(t+\Delta t)-\frac{\Delta t}{2} A c(t+\Delta t) & =(I+\Delta t / 2 A) c(t)+\Delta t B f(t) \\
c(t+\Delta t) & =(I-\Delta t / 2 A)^{-1}(I+\Delta t / 2 A) c(t)+\Delta t(I-\Delta t / 2 A)^{-1} B f(t)。
\end{aligned}
$$

广义双线性变换（GBT）。这种方法 [77] 通过对端点的加权平均 $\Delta t[(1-\alpha) f(t, c(t))+\alpha f(t+\Delta t, c(t+\Delta t))]$ 来近似方程（12）的 RHS，其中 $\alpha \in[0,1]$。对于线性 ODE，我们再次得到线性方程和更新：

$$
\begin{aligned}
c(t+\Delta t)-\Delta t \alpha A c(t+\Delta t) & =(I+\Delta t(1-\alpha) A) c(t)+\Delta t B f(t) \\
c(t+\Delta t) & =(I-\Delta t \alpha A)^{-1}(I+\Delta t(1-\alpha) A) c(t)+\Delta t(I-\Delta t \alpha A)^{-1} B f(t)。
\end{aligned}
$$

GBT 推广了上述提到的三种方法：前向欧拉对应于 $\alpha=0$，反向欧拉对应于 $\alpha=1$，而双线性对应于 $\alpha=1/2$。

我们还注意到另一种方法称为零阶保持（ZOH）[23]，它专门用于线性 ODE。假定在 $t$ 和 $t+\Delta t$ 之间的常量输入 $f$ 下计算方程（12）的 RHS。这将得到更新 $c(t+\Delta t)=e^{\Delta t A} c(t)+\left(\int_{\tau=0}^{\Delta t} e^{\tau A} \mathrm{~d} \tau\right) B f(t)$。如果 $A$ 可逆，则可以简化为 $c(t+\Delta t)=e^{\Delta t A} c(t)+A^{-1}\left(e^{\Delta t A}-I\right) B f(t)$​。

HiPPO-LegS 对于离散化步长的不变性。在 HiPPO-LegS 的情况下，我们有一个形式为 $\frac{d}{d t} c(t)=\frac{1}{t} A c(t)+\frac{1}{t} B f(t)$ 的线性 ODE。将 GBT 离散化（它泛化了前向/反向欧拉和双线性）应用于这个线性 ODE，我们得到：

$$
\begin{aligned}
c(t+\Delta t)-\Delta t \alpha \frac{1}{t+\Delta t} A c(t+\Delta t) & =\left(I+\Delta t(1-\alpha) \frac{1}{t} A\right) c(t)+\Delta t \frac{1}{t} B f(t) \\
c(t+\Delta t) & =\left(I-\frac{\Delta t}{t+\Delta t} \alpha A\right)^{-1}\left(I+\frac{\Delta t}{t}(1-\alpha) A\right) c(t)+\frac{\Delta t}{t}\left(I-\frac{\Delta t}{t+\Delta t} \alpha A\right)^{-1} B
\end{aligned}
$$

我们强调，这个系统对于离散化步长 $\Delta t$ 是不变的。实际上，如果 $c^{(k)}:=c(k \Delta t)$ 和 $f_k:=f(k \Delta t)$，那么我们有递推关系

$$
c^{(k+1)}=\left(I-\frac{1}{k+1} \alpha A\right)^{-1}\left(I+\frac{1}{k}(1-\alpha) A\right) c^{(k)}

+\frac{1}{k}\left(I-\frac{1}{k+1} \alpha A\right)^{-1} B f_k，
$$

这不依赖于 $\Delta t$。

消融：不同离散化方法的比较为了理解离散化中的近似误差的影响，在图 4 中，我们展示了 HiPPO-LegS 更新在函数逼近（附录 F.8）中的绝对误差，对于不同的离散化方法：前向欧拉、反向欧拉和双线性。双线性方法通常提供足够精确的近似。我们将使用双线性作为实验中 LegS 更新的离散化方法。

![image.png](https://raw.githubusercontent.com/aletolia/Pictures/main/202403301506186.png)

图 4：不同离散化方法的绝对误差。前向和反向欧拉通常不太准确，而双线性方法产生了更准确的近似。

### C 通用 HiPPO 框架

我们详细介绍通用的 HiPPO 框架，如在第 2 节中所述。我们还将其推广以包括除多项式之外的其他基。

给定时间变化的测度族 $\mu^{(t)}$，支撑在 $(-\infty, t]$ 上，一系列基函数 $\mathcal{G}=\operatorname{span}\left\{g_n^{(t)}\right\}_{n \in[N]}$，以及连续函数 $f: \mathbb{R}_{\geq 0} \rightarrow \mathbb{R}$，HiPPO 定义了一个算子，将 $f$ 映射到最优投影系数 $c: \mathbb{R}_{\geq 0} \rightarrow \mathbb{R}^N$，使得

$$
g^{(t)}:=\operatorname{argmin}_{g \in \mathcal{G}}\left\|f_{\leq t}-g\right\|_{\mu^{(t)}}, \quad \text { and } \quad g^{(t)}=\sum_{n=0}^{N-1} c_n(t) g_n^{(t)} .
$$

第一步指的是定义 1 中的 proj $_t$ 算子，第二步是 coef $t$ 算子。我们着重讨论系数 $c(t)$ 具有线性常微分方程 (ODE) 形式的情况，满足 $\frac{d}{d t} c(t)=A(t) c(t)+B(t) f(t)$，其中 $A(t) \in \mathbb{R}^{N \times N}$，$B(t) \in \mathbb{R}^{N \times 1}$。

我们首先在附录 C.1 中更详细地描述了 hippo 算子（测度和基）。我们在附录 C.2 中定义了投影 proj $_t$ 和系数 coef $t$ 算子。然后我们给出了一般的策略来计算这些系数 $c(t)$，通过导出控制系数动态的微分方程（附录 C.3）。最后我们讨论如何将连续的 hippo 算子转换为一个离散的算子，可以应用于序列数据（附录 C.4）。

#### C.1 通用 HiPPO 框架

我们在这里更详细地描述和解释 HiPPO 的组成部分。回顾一下高层目标是在线函数逼近；这需要一组有效的逼近以及逼近质量的概念。

逼近测度：在每个 $t$，逼近质量是指相对于支持在 $(-\infty, t]$ 上的测度 $\mu^{(t)}$。我们寻找一些多项式 $g^{(t)}$，其次数最多为 $N-1$，使得误差 $\left\|f_{x \leq t}-g^{(t)}\right\|_{L_2\left(\mu^{(t)}\right)}$ 最小化。直观地说，这个测度 $\mu^{(t)}$ 决定了过去的每个时间点的权重。为简单起见，我们假设测度 $\mu^{(t)}$ 在其定义域内以及时间上足够平滑；**==特别地，它们具有密度 $\omega(t, x):=\frac{\mathrm{d} \mu^{(t)}}{\mathrm{d} \lambda}(x)$，其中勒贝格测度 $\mathrm{d} \lambda(x):=\mathrm{d} x$，使得 $\omega$ 几乎处处 $C^1$。因此，对 $\mathrm{d} \mu^{(t)}(x)$ 进行积分可以重写为对 $\omega(t, x) \mathrm{d} x$ 进行积分。==**

为了简化，我们还假设测度 $\mu^{(t)}$ 被归一化为概率测度；任意缩放不会影响最优投影。

**==正交多项式基：让 $\left\{P_n\right\}_{n \in \mathbb{N}}$ 表示相对于某个基测度 $\mu$ 的一系列正交多项式==**。类似地，定义 $\left\{P_n^{(t)}\right\}_{n \in \mathbb{N}}$ 为相对于时间变化的测度 $\mu^{(t)}$ 的一系列正交多项式。令 $p_n^{(t)}$ 为 $P_n^{(t)}$ 的归一化版本（即，具有单位范数），并定义

$$
p_n(t, x)=p_n^{(t)}(x)。\tag{14}
$$

注意，$P_n^{(t)}$ 不需要被归一化，而 $p_n^{(t)}$ 需要。

**==倾斜的测度和基础：我们的目标只是存储函数的压缩表示，可以使用任何基，不一定是正交多项式。对于任意缩放函数==**

$$
\chi(t, x)=\chi^{(t)}(x)，\tag{15}
$$

函数 $p_n(x) \chi(x)$ 相对于每个时间 $t$ 上以概率密度 $\omega / \chi^2$ 正交。因此，我们可以选择这个替代的基和测度来执行投影。

**==为了正式化这种倾斜的作用，定义 $\nu^{(t)}$ 为密度与 $\omega^{(t)} /\left(\chi^{(t)}\right)^2$ 成比例的归一化测度==**。我们将计算归一化测度和它的标准正交基。令

$$
\zeta(t)=\int \frac{\omega}{\chi^2}=\int \frac{\omega^{(t)}(x)}{\left(\chi^{(t)}(x)\right)^2} \mathrm{~d} x\tag{16}
$$

为归一化常数，以便 $\nu^{(t)}$ 具有密度 $\frac{\omega^{(t)}}{\zeta(t)\left(\chi^{(t)}\right)^2}$。如果 $\chi(t, x)=1$（没有倾斜），则该常数为 $\zeta(t)=1$。一般来说，我们假设 $\zeta$ 对所有 $t$ 是常数；如果不是，它可以直接折叠到 $\chi$ 中。

接下来，注意（为简洁起见，在积分内部省略对 $x$ 的依赖）：

$$
\begin{aligned}
\left\|\zeta(t)^{\frac{1}{2}} p_n^{(t)} \chi^{(t)}\right\|_{\nu^{(t)}}^2 & =\int\left(\zeta(t)^{\frac{1}{2}} p_n^{(t)} \chi^{(t)}\right)^2 \frac{\omega^{(t)}}{\zeta(t)\left(\chi^{(t)}\right)^2} \\
& =\int\left(p_n^{(t)}\right)^2 \omega^{(t)} \\
& =\left\|p_n^{(t)}\right\|_{\mu^{(t)}}^2=1 .
\end{aligned}
$$

因此，我们定义 $\nu^{(t)}$ 的正交基：

$$
g_n^{(t)}=\lambda_n \zeta(t)^{\frac{1}{2}} p_n^{(t)} \chi^{(t)}, \quad n \in \mathbb{N} .\tag{17}
$$

我们让基的每个元素都乘以一个 $\lambda_n$ 标量，出于很快要讨论的原因，因为任意缩放不会改变正交性：

$$
\left\langle g_n^{(t)}, g_m^{(t)}\right\rangle_{\nu^{(t)}}=\lambda_n^2 \delta_{n, m}
$$

注意当 $\lambda_n= \pm 1$ 时，基 $\left\{g_n^{(t)}\right\}$ 是关于测度 $\nu^{(t)}$ 正交的标准正交基，对于每个时间 $t$。按惯例记号，让 $g_n(t, x):=g_n^{(t)}(x)$。我们只在拉盖尔（附录 D.2）和切比雪夫（附录 D.5）的情况下使用这种倾斜。

注意在 $\chi=1$（即没有倾斜）的情况下，我们还有 $\zeta=1$ 和 $g_n=\lambda_n p_n$（对于所有的 $t, x$）。

#### C.2 投影和系数

在给定测度和基函数的选择后，我们接下来看看如何计算系数 $c(t)$。

输入：函数

我们有一个在线观察到的 $C^1$ 光滑函数 $f:[0, \infty) \rightarrow \mathbb{R}$，我们希望在每个时间 $t$ 维护其历史的压缩表示 $f(x)_{\leq t}=f(x)_{x \leq t}$。

输出：近似系数

函数 $f$ 可以通过存储其相对于基 $\left\{g_n\right\}_{n<N}$ 的系数来进行近似。例如，在没有倾斜 $\chi=1$ 的情况下，这编码了 $f$ 的低于 $N$ 阶多项式逼近。特别地，在时间 $t$，我们希望将 $f_{\leq t}$ 表示为多项式 $g_n^{(t)}$ 的线性组合。由于 $g_n^{(t)}$ 关于由 $\langle\cdot, \cdot\rangle_{\nu^{(t)}}$ 定义的希尔伯特空间正交，计算系数就足够了：

$$
\begin{aligned}
c_n(t) & =\left\langle f_{\leq t}, g_n^{(t)}\right\rangle_{\nu^{(t)}} \\
& =\int f g_n^{(t)} \frac{\omega^{(t)}}{\zeta(t)\left(\chi^{(t)}\right)^2} \\
& =\zeta(t)^{-\frac{1}{2}} \lambda_n \int f p_n^{(t)} \frac{\omega^{(t)}}{\chi^{(t)}} .
\end{aligned}\tag{18}
$$

重构

在任意时间 $t$，$f_{\leq t}$ 可以被明确地重构为

$$
\begin{aligned}
f_{\leq t} \approx g^{(t)} & :=\sum_{n=0}^{N-1}\left\langle f_{\leq t}, g_n^{(t)}\right\rangle_{\nu^{(t)}} \frac{g_n^{(t)}}{\left\|g_n^{(t)}\right\|_{\nu^{(t)}}^2} \\
& =\sum_{n=0}^{N-1} \lambda_n^{-2} c_n(t) g_n^{(t)} \\
& =\sum_{n=0}^{N-1} \lambda_n^{-1} \zeta^{\frac{1}{2}} c_n(t) p_n^{(t)} \chi^{(t)} .
\end{aligned}\tag{19}
$$

方程 $(19)$ 是 proj $_t$ 算子；给定测度和基参数，它定义了 $f_{\leq t}$ 的最优逼近。

#### C.3 系数动态：Hippo 算子

为了满足端到端模型对输入函数 $f(t)$ 的需求，系数 $c(t)$ 足以编码关于 $f$ 历史的信息并允许在线预测。因此，将 $c(t)$ 定义为方程（18）中的 $c_n(t)$ 向量，我们的重点将放在如何从输入函数 $f: \mathbb{R}_{\geq 0} \rightarrow \mathbb{R}$ 计算函数 $c: \mathbb{R}_{\geq 0} \rightarrow \mathbb{R}^N$​ 上。

在我们的框架中，我们将把这些系数视为动态系统随时间演化而计算。对方程（18）进行微分，得到：
$$
\begin{aligned}
\frac{d}{d t} c_n(t)= & \zeta(t)^{-\frac{1}{2}} \lambda_n \int f(x)\left(\frac{\partial}{\partial t} p_n(t, x)\right) \frac{\omega}{\chi}(t, x) \mathrm{d} x \\
& +\int f(x)\left(\zeta^{-\frac{1}{2}} \lambda_n p_n(t, x)\right)\left(\frac{\partial}{\partial t} \frac{\omega}{\chi}(t, x)\right) \mathrm{d} x .
\end{aligned}\tag{20}
$$

在这里，我们利用了 $\zeta$ 对于所有 $t$ 都是常数的假设。

令 $c(t) \in \mathbb{R}^{N-1}$ 表示所有系数 $\left(c_n(t)\right)_{0 \leq n<N}$ 的向量。

关键思想是，如果 $\frac{\partial}{\partial t} P_n$ 和 $\frac{\partial}{\partial t} \frac{\omega}{\chi}$ 具有可以与多项式 $P_k$ 相关联的闭式，那么可以为 $c(t)$ 写出一个普通微分方程。这使得这些系数 $c(t)$ 和因此最优多项式逼近可以在线计算。由于 $\frac{d}{d t} P_n^{(t)}$ 是一个关于 $x$ 的 $n-1$ 次多项式，可以将其写成 $P_0, \ldots, P_{n-1}$ 的线性组合，因此方程（20）的第一项是 $c_0, \ldots, c_{n-1}$ 的线性组合。对于许多权重函数 $\omega$，我们可以找到缩放函数 $\chi$，使得 $\frac{\partial}{\partial t} \frac{\omega}{\chi}$ 也可以用 $\frac{\omega}{\chi}$ 本身来表示，因此在这些情况下，方程（20）的第二项也是 $c_0, \ldots, c_{N-1}$ 和输入 $f$ 的线性组合。因此，这通常导致 $c(t)$​​ 的封闭形式线性常微分方程。

##### 归一化动态

我们定义自由参数 $\lambda_n$ 的目的有三个。

1. 首先，注意标准正交基不是唯一的，每个元素可以乘以 $a \pm 1$ 因子。
2. 第二，选择 $\lambda_n$ 可以帮助简化推导。
3. 第三，虽然选择 $\lambda_n= \pm 1$ 将是我们的默认选择，因为投影到标准正交基是最合理的，但 LMU 使用了不同的缩放。附录 [D.1] 将通过为 LegT 测度选择不同的 $\lambda_n$ 来恢复 LMU。

假设方程（20）简化为形式如下的动态系统：

$$
\frac{d}{d t} c(t)=-A(t) c(t)+B(t) f(t) .
$$

然后，令 $\Lambda=\operatorname{diag}_{n \in[N]}\left\{\lambda_n\right\}$，

$$
\frac{d}{d t} \Lambda^{-1} c(t)=-\Lambda^{-1} A(t) \Lambda \Lambda^{-1} c(t)+\Lambda^{-1} B(t) f(t) .
$$

因此，如果我们重新参数化系数 $\left(\Lambda^{-1} c(t) \rightarrow c(t)\right)$，那么投影到标准正交基上的归一化系数将满足动态和相关重构：

$$
\begin{aligned}
\frac{d}{d t} c(t) & =-\left(\Lambda^{-1} A(t) \Lambda\right) c(t)+\left(\Lambda^{-1} B(t)\right) f(t) \\
f_{\leq t} \approx g^{(t)} & =\sum_{n=0}^{N-1} \zeta^{\frac{1}{2}} c_n(t) p_n^{(t)} \chi^{(t)}
\end{aligned}\tag{21,22}
$$

这些是 hippo 和 proj $_t$ 算子。

#### C.4 离散化

如此定义，hippo 是连续函数的映射。然而，由于 hippo 定义了一个系数动态的闭式常微分方程，标准的常微分方程离散化方法（附录 B.3）可以应用于将其转换为离散的内存更新。因此，我们重载这些算子，即 hippo 要么定义成形式为

$$
\frac{d}{d t} c(t)=A(t) c(t)+B(t) f(t)
$$

或者是一个递归

$$
c_t=A_t c_{t-1}+B_t f_t,
$$

这可以根据上下文进行清晰的选择。

附录 F.5 通过将（20）和（19）应用于近似合成函数来验证了该框架。

![image.png](https://raw.githubusercontent.com/aletolia/Pictures/main/202403301516567.png)

 图 5：HiPPO 测度的说明 在时间 $t_0$，函数 $f(x)_{x \leq t_0}$ 的历史通过关于测度 $\mu^{\left(t_0\right)}$ 的多项式逼近进行总结（蓝色），类似地，在时间 $t_1$（紫色）也是如此。（左）平移的勒让德测度（LegT）在窗口 $[t-\theta, t]$ 中分配权重。对于小的 $t$，$\mu^{(t)}$ 支持在 $x<0$ 的区域，而 $f$ 未定义。当 $t$ 很大时，测度不支持接近 0 的区域，导致对 $f$ 的投影忘记了函数的开头。（中）平移的拉盖尔（LagT）测度以指数方式衰减过去。它不会忘记，但也会在 $x<0$ 上分配权重。（右）缩放的勒让德测度（LegS）均匀地加权整个历史 $[0, t]$。

### D HiPPO 投影算子的推导

我们推导了与在第 2.3 节中介绍的平移的勒让德（LegT）和平移的拉盖尔（LagT）测度以及在第 3 节中介绍的缩放的勒让德（LegS）测度相关的记忆更新，以展示框架的普适性。为了显示框架的普适性，我们还推导了使用傅里叶基函数（恢复傅里叶循环单元 [79]）和切比雪夫基函数的记忆更新。

绝大部分工作已经通过建立投影框架完成，证明只需按照附录 C 中概述的技术大纲进行。特别地，系数的定义（18）和重构（19）不会改变，我们只考虑如何计算系数动态（20）。

对于每种情况，我们遵循以下一般步骤：
- **测度和基函数**：定义测度 $\mu^{(t)}$ 或权重 $\omega(t, x)$，以及基函数 $p_n(t, x)$。
- **导数**：计算测度和基函数的导数。
- **系数动态**：将它们代入系数动态（方程 (20)）中，推导描述如何计算系数 $c(t)$ 的常微分方程。
- **重构**：提供重构函数 $f_{\leq t}$ 的完整公式，这是在此测度和基础下的最优投影。

在附录 D.1 和 D.2 中的推导证明了定理 1，而在附录 D.3 中的推导证明了定理 2。附录 D.4 和 D.5 展示了基于傅里叶的基函数的额外结果。图 5 说明了当我们使用勒让德和拉盖尔多项式作为基础时，对比了我们的主要时间变化测度族 $\mu^{(t)}$ 的整体框架。

#### D.1 平移的勒让德测度的推导（HiPPO-LegT）

该测度固定了一个窗口长度 $\theta$，并将其沿时间滑动。

##### 测度和基函数

我们使用在区间 $[t-\theta, t]$ 上支持的均匀权重函数，并选择勒让德多项式 $P_n(x)$ 作为基函数，将其从 $[-1,1]$ 平移至 $[t-\theta, t]$：

$$
\begin{aligned}
\omega(t, x) & =\frac{1}{\theta} \mathbb{I}_{[t-\theta, t]} \\
p_n(t, x) & =(2 n+1)^{1 / 2} P_n\left(\frac{2(x-t)}{\theta}+1\right) \\
g_n(t, x) & =\lambda_n p_n(t, x) .
\end{aligned}
$$

这里，我们没有使用倾斜，所以 $\chi=1$ 且 $\zeta=1$（方程 (15) 和 (16)）。我们暂时不指定 $\lambda_n$。
在端点处，这些基函数满足

$$
\begin{aligned}
g_n(t, t) & =\lambda_n(2 n+1)^{\frac{1}{2}} \\
g_n(t, t-\theta) & =\lambda_n(-1)^n(2 n+1)^{\frac{1}{2}} .
\end{aligned}
$$

##### 导数

测度的导数是

$$
\frac{\partial}{\partial t} \omega(t, x)=\frac{1}{\theta} \delta_t-\frac{1}{\theta} \delta_{t-\theta} .
$$

勒让德多项式的导数可以表示为其他勒让德多项式的线性组合（参见附录 B.1.1）。

$$
\begin{aligned}
\frac{\partial}{\partial t} g_n(t, x) & =\lambda_n(2 n+1)^{\frac{1}{2}} \cdot \frac{-2}{\theta} P_n^{\prime}\left(\frac{2(x-t)}{\theta}+1\right) \\
& =\lambda_n(2 n+1)^{\frac{1}{2}} \frac{-2}{\theta}\left[(2 n-1) P_{n-1}\left(\frac{2(x-t)}{\theta}+1\right)+(2 n-5) P_{n-3}\left(\frac{2(x-t)}{\theta}+1\right)+\ldots\right] \\
& =-\lambda_n(2 n+1)^{\frac{1}{2}} \frac{2}{\theta}\left[\lambda_{n-1}^{-1}(2 n-1)^{\frac{1}{2}} g_{n-1}(t, x)+\lambda_{n-3}^{-1}(2 n-3)^{\frac{1}{2}} g_{n-3}(t, x)+\ldots\right] .
\end{aligned}
$$

我们在这里使用了方程 (7)。

##### 滑动近似

作为 LegT 测度的特例，我们需要考虑由于滑动窗口测度的特性而产生的近似。
在下一节分析 $\frac{d}{d t} c(t)$ 时，我们将需要使用值 $f(t-\theta)$。然而，在时间 $t$，此输入不再可用。相反，我们需要依赖于函数的压缩表示：根据重构方程（19），如果近似到目前为止是成功的，我们应该有

$$
\begin{aligned}
& f_{\leq t}(x) \approx \sum_{k=0}^{N-1} \lambda_k^{-1} c_k(t)(2 k+1)^{\frac{1}{2}} P_k\left(\frac{2(x-t)}{\theta}+1\right) \\
& f(t-\theta) \approx \sum_{k=0}^{N-1} \lambda_k^{-1} c_k(t)(2 k+1)^{\frac{1}{2}}(-1)^k .
\end{aligned}
$$

##### 系数动态

我们准备推导系数动态方程。

将测度和基函数的导数代入方程 (20) 中得到

$$
\begin{aligned}
\frac{d}{d t} c_n(t)=\int & f(x)\left(\frac{\partial}{\partial t} g_n(t, x)\right) \omega(t, x) \mathrm{d} x \\
& +\int f(x) g_n(t, x)\left(\frac{\partial}{\partial t} \omega(t, x)\right) \mathrm{d} x \\
=- & \lambda_n(2 n+1)^{\frac{1}{2}} \frac{2}{\theta}\left[\lambda_{n-1}^{-1}(2 n-1)^{\frac{1}{2}} c_{n-1}(t)+\lambda_{n-3}^{-1}(2 n-5)^{\frac{1}{2}} c_{n-3}(t)+\ldots\right] \\
& +\frac{1}{\theta} f(t) g_n(t, t)-\frac{1}{\theta} f(t-\theta) g_n(t, t-\theta) \\
\approx- & \frac{\lambda_n}{\theta}(2 n+1)^{\frac{1}{2}} \cdot 2\left[(2 n-1)^{\frac{1}{2}} \frac{c_{n-1}(t)}{\lambda_{n-1}}+(2 n-5)^{\frac{1}{2}} \frac{c_{n-3}(t)}{\lambda_{n-3}}+\ldots\right] \\
& +(2 n+1)^{\frac{1}{2}} \frac{\lambda_n}{\theta} f(t)-(2 n+1)^{\frac{1}{2}} \frac{\lambda_n}{\theta}(-1)^n \sum_{k=0}^{N-1}(2 k+1)^{\frac{1}{2}} \frac{c_k(t)}{\lambda_k}(-1)^k \\
=- & \frac{\lambda_n}{\theta}(2 n+1)^{\frac{1}{2}} \cdot 2\left[(2 n-1)^{\frac{1}{2}} \frac{c_{n-1}(t)}{\lambda_{n-1}}+(2 n-5)^{\frac{1}{2}} \frac{c_{n-3}(t)}{\lambda_{n-3}}+\ldots\right] \\
& -(2 n+1)^{\frac{1}{2}} \frac{\lambda_n}{\theta} \sum_{k=0}^{N-1}(-1)^{n-k}(2 k+1)^{\frac{1}{2}} \frac{c_k(t)}{\lambda_k}+(2 n+1)^{\frac{1}{2}} \frac{\lambda_n}{\theta} f(t) \\
=- & \frac{\lambda_n}{\theta}(2 n+1)^{\frac{1}{2}} \sum_{k=0}^{N-1} M_{n k}(2 k+1)^{\frac{1}{2}} \frac{c_k(t)}{\lambda_k}+(2 n+1)^{\frac{1}{2}} \frac{\lambda_n}{\theta} f(t),
\end{aligned}
$$

其中

$$
M_{n k}=\left\{\begin{array}{ll}
1 & \text { if } k \leq n \\
(-1)^{n-k} & \text { if } k \geq n
\end{array} .\right.
$$

现在我们考虑 $\lambda_n$ 的两种实例化。第一种是更自然的 $\lambda_n=1$，这将使 $g_n$ 成为正交基。我们得到

$$
\begin{aligned}
\frac{d}{d t} c(t) & =-\frac{1}{\theta} A c(t)+\frac{1}{\theta} B f(t) \\
A_{n k} & =(2 n+1)^{\frac{1}{2}}(2 k+1)^{\frac{1}{2}} \begin{cases}1 & \text { if } k \leq n \\
(-1)^{n-k} & \text { if } k \geq n\end{cases} \\
B_n & =(2 n+1)^{\frac{1}{2}} .
\end{aligned}
$$

第二种情况取 $\lambda_n=(2 n+1)^{\frac{1}{2}}(-1)^n$。这产生了

$$
\begin{aligned}
\frac{d}{d t} c(t) & =-\frac{1}{\theta} A c(t)+\frac{1}{\theta} B f(t) \\
A_{n k} & =(2 n+1) \begin{cases}(-1)^{n-k} & \text { if } k \leq n \\
1 & \text { if } k \geq n\end{cases} \\
B_n & =(2 n+1)(-1)^n .
\end{aligned}
$$

这正是 LMU 的更新方程。根据方程 (19)，在每个时间 $t$，我们有

$$
f(x) \approx g^{(t)}(x)=\sum_n \lambda_n^{-1} c_n(t)(2 n+1)^{\frac{1}{2}} P_n\left(\frac{2(x-t)}{\theta}+1\right) .
$$

#### D.2 翻译的拉盖尔（HiPPO-LagT）的推导

我们考虑基于广义拉盖尔多项式的测度。对于固定的 $\alpha \in \mathbb{R}$，这些多项式 $L^{(\alpha)}(t-x)$ 关于区间 $[0, \infty)$ 上的测度 $x^\alpha e^{-x}$ 是正交的（参见附录 B.1.2）。此推导将涉及到根据另一个参数 $\beta$ 对测度进行倾斜。

HiPPO-LagT 的定理 1 的结果是 $\alpha=0, \beta=1$ 的情况，对应于基本拉盖尔多项式且没有倾斜。

##### 测度和基函数

我们将广义拉盖尔权函数和多项式从 $[0, \infty)$ 翻转并平移到 $(-\infty, t]$。使用方程 (9) 找到归一化。

$$
\begin{aligned}
\omega(t, x) & = \begin{cases}(t-x)^\alpha e^{x-t} & \text { if } x \leq t \\
0 & \text { if } x>t\end{cases} \\
& =(t-x)^\alpha e^{-(t-x)} \mathbb{I}_{(-\infty, t]} \\
p_n(t, x) & =\frac{\Gamma(n+1)^{\frac{1}{2}}}{\Gamma(n+\alpha+1)^{\frac{1}{2}}} L_n^{(\alpha)}(t-x)
\end{aligned}
$$

##### 倾斜的测度

我们选择以下倾斜 $\chi$

$$
\chi(t, x)=(t-x)^\alpha \exp \left(-\frac{1-\beta}{2}(t-x)\right) \mathbb{I}_{(-\infty, t]}
$$

对于某个固定的 $\beta \in \mathbb{R}$。归一化是（对所有 $t$ 都是常数）

$$
\begin{aligned}
\zeta & =\int \frac{\omega}{\chi^2}=\int(t-x)^{-\alpha} e^{-\beta(t-x)} \mathbb{I}_{(-\infty, t]} \mathrm{d} x \\
& =\Gamma(1-\alpha) \beta^{\alpha-1},
\end{aligned}
$$

因此倾斜的测度具有密度

$$
\zeta(t)^{-1} \frac{\omega^{(t)}}{\left(\chi^{(t)}\right)^2}=\Gamma(1-\alpha)^{-1} \beta^{1-\alpha}(t-x)^{-\alpha} \exp (-\beta(t-x)) \mathbb{I}_{(-\infty, t]} .
$$

我们选择

$$
\lambda_n=\frac{\Gamma(n+\alpha+1)^{\frac{1}{2}}}{\Gamma(n+1)^{\frac{1}{2}}}
$$

作为广义拉盖尔多项式 $L_n^{(\alpha)}$ 的范数，这样 $\lambda_n p_n^{(t)}=L_n^{(\alpha)}(t-x)$，并且（根据方程 (17)）$\nu^{(t)}$ 的基函数是

$$
\begin{aligned}
g_n^{(t)} & =\lambda_n \zeta^{\frac{1}{2}} p_n^{(t)} \chi^{(t)} \\
& =\zeta^{\frac{1}{2}} \chi^{(t)} L_n^{(\alpha)}(t-x)
\end{aligned}\tag{23}
$$

##### 导数

我们首先计算密度比

$$
\frac{\omega}{\chi}(t, x)=\exp \left(-\frac{1+\beta}{2}(t-x)\right) \mathbb{I}_{(-\infty, t]} .
$$

以及它的导数

$$
\frac{\partial}{\partial t} \frac{\omega}{\chi}(t, x)=-\left(\frac{1+\beta}{2}\right) \frac{\omega}{\chi}(t, x)+\exp \left(-\left(\frac{1+\beta}{2}\right)(t-x)\right) \delta_t .
$$

拉盖尔多项式的导数可以表示为其他拉盖尔多项式的线性组合（参见附录 B.1.2）。

$$
\begin{aligned}
\frac{\partial}{\partial t} \lambda_n p_n(t, x) & =\frac{\partial}{\partial t} L_n^{(\alpha)}(t-x) \\
& =-L_0^{(\alpha)}(t-x)-\cdots-L_{n-1}^{(\alpha)}(t-x) \\
& =-\lambda_0 p_0(t, x)-\cdots-\lambda_{n-1} p_{n-1}(t, x)
\end{aligned}
$$


##### 系数动态

将这些导数代入方程 (20)（从微分系数方程 (18) 中获得），其中为方便起见，我们忽略对 $x$ 的依赖：

$$
\begin{aligned}
\frac{d}{d t} c_n(t)= & \zeta^{-\frac{1}{2}} \int f \cdot\left(\frac{\partial}{\partial t} \lambda_n p_n^{(t)}\right) \frac{\omega^{(t)}}{\chi^{(t)}} \\
& +\int f \cdot\left(\zeta^{-\frac{1}{2}} \lambda_n p_n^{(t)}\right)\left(\frac{\partial}{\partial t} \frac{\omega^{(t)}}{\chi^{(t)}}\right) \\
= & -\sum_{k=0}^{n-1} \int f \cdot\left(\zeta^{-\frac{1}{2}} \lambda_k p_k^{(t)} \chi^{(t)}\right) \frac{\omega^{(t)}}{\left(\chi^{(t)}\right)^2} \\
& -\left(\frac{1+\beta}{2}\right) \int f \cdot\left(\zeta^{-\frac{1}{2}} \lambda_n p_n^{(t)}\right) \frac{\omega^{(t)}}{\chi^{(t)}}+f(t) \cdot \zeta^{-\frac{1}{2}} L_n^{(\alpha)}(0) \\
= & -\sum_{k=0}^{n-1} c_k(t)-\left(\frac{1+\beta}{2}\right) c_n(t)+\Gamma(1-\alpha)^{-\frac{1}{2}} \beta^{\frac{1-\alpha}{2}}\left(\begin{array}{c}
n+\alpha \\
n
\end{array}\right) f(t) .
\end{aligned}
$$

然后我们得到

$$
\begin{aligned}
\frac{d}{d t} c(t) & =-A c(t)+B f(t) \\
A & =\left[\begin{array}{cccc}
\frac{1+\beta}{2} & 0 & \ldots & 0 \\
1 & \frac{1+\beta}{2} & \ldots & 0 \\
\vdots & & \ddots & \\
1 & 1 & \ldots & \frac{1+\beta}{2}
\end{array}\right] \\
B & =\zeta^{-\frac{1}{2}} \cdot\left[\begin{array}{c}
\left(\begin{array}{c}
\alpha \\
0
\end{array}\right) \\
\vdots \\
\left(\begin{array}{c}
N-1+\alpha \\
N-1
\end{array}\right)
\end{array}\right]
\end{aligned}\tag{24}
$$

##### 重构

根据方程 (19)，在每个时间 $t$，对于 $x \leq t$，

$$
\begin{aligned}
f(x) \approx g^{(t)}(x) & =\sum_{n=0}^{N-1} \lambda_n^{-1} \zeta^{\frac{1}{2}} c_n(t) p_n^{(t)} \chi^{(t)} \\
& =\sum_n \frac{n !}{(n+\alpha) !} \zeta^{\frac{1}{2}} c_n(t) \cdot L_n^{(\alpha)}(t-x) \cdot(t-x)^\alpha e^{\left(\frac{\beta-1}{2}\right)(t-x) .}
\end{aligned}
$$

##### 标准化动态

最后，遵循方程 (21) 和 (22) 将这些转换为对归一化（概率）测度 $\nu^{(t)}$ 的正交基的动态，得到以下 hippo 运算符

$$
\begin{aligned}
\frac{d}{d t} c(t) & =-A c(t)+B f(t) \\
A & =-\Lambda^{-1}\left[\begin{array}{cccc}
\frac{1+\beta}{2} & 0 & \ldots & 0 \\
1 & \frac{1+\beta}{2} & \ldots & 0 \\
\vdots & & \ddots & \\
1 & 1 & \ldots & \frac{1+\beta}{2}
\end{array}\right] \Lambda \\
B & =\Gamma(1-\alpha)^{-\frac{1}{2}} \beta^{\frac{1-\alpha}{2}} \cdot \Lambda^{-1}\left[\begin{array}{c}
\left(\begin{array}{c}
\alpha \\
0
\end{array}\right) \\
\vdots \\
\left(\begin{array}{c}
N-1+\alpha \\
N-1
\end{array}\right)
\end{array}\right] \\
\Lambda & =\operatorname{diag}\left\{\frac{\Gamma(n+\alpha+1)^{\frac{1}{2}}}{\Gamma(n+1)^{\frac{1}{2}}}\right\}
\end{aligned}
$$

相应地，有一个 `proj $_t$` 运算符：

$$
\begin{aligned}
f(x) \approx g^{(t)}(x) & =\Gamma(1-\alpha)^{\frac{1}{2}} \beta^{-\frac{1-\alpha}{2}} \sum_n c_n(t) \cdot \frac{\Gamma(n+1)^{\frac{1}{2}}}{\Gamma(n+\alpha+1)^{\frac{1}{2}}} \cdot L_n^{(\alpha)}(t-x) \cdot(t-x)^\alpha e^{\left(\frac{\beta-1}{2}\right)(t-x)} .
\end{aligned}
$$

#### D.3 缩放勒让德（HiPPO-LegS）的推导

如第 3 节所讨论的那样，缩放勒让德是我们唯一使用具有不同宽度的测度的方法。

测度与基础 我们在以下情况下实例化框架

$$
\begin{aligned}
\omega(t, x) & =\frac{1}{t} \mathbb{I}_{[0, t]} \\
g_n(t, x) & =p_n(t, x)=(2 n+1)^{\frac{1}{2}} P_n\left(\frac{2 x}{t}-1\right)
\end{aligned}\tag{27}
$$

这里，$P_n$ 是基本的勒让德多项式（附录 B.1.1）。我们不使用倾斜，即 $\chi(t, x)=1$，$\zeta(t)=1$，$\lambda_n=1$，以便函数 $g_n(t, x)$ 形成正交基。

导数 我们首先对测度和基进行微分：

$$
\begin{aligned}
\frac{\partial}{\partial t} \omega(t, \cdot) & =-t^{-2} \mathbb{I}_{[0, t]}+t^{-1} \delta_t=t^{-1}\left(-\omega(t)+\delta_t\right) \\
\frac{\partial}{\partial t} g_n(t, x) & =-(2 n+1)^{\frac{1}{2}} 2 x t^{-2} P_n^{\prime}\left(\frac{2 x}{t}-1\right) \\
& =-(2 n+1)^{\frac{1}{2}} t^{-1}\left(\frac{2 x}{t}-1+1\right) P_n^{\prime}\left(\frac{2 x}{t}-1\right) .
\end{aligned}\tag{28}
$$

现在定义 $z=\frac{2 x}{t}-1$ 为简便起见，并应用勒让德多项式的导数性质（方程（8））。

$$
\begin{aligned}
\frac{\partial}{\partial t} g_n(t, x) & =-(2 n+1)^{\frac{1}{2}} t^{-1}(z+1) P_n^{\prime}(z) \\
& =-(2 n+1)^{\frac{1}{2}} t^{-1}\left[n P_n(z)+(2 n-1) P_{n-1}(z)+(2 n-3) P_{n-2}(z)+\ldots\right] \\
& =-t^{-1}(2 n+1)^{\frac{1}{2}}\left[n(2 n+1)^{-\frac{1}{2}} g_n(t, x)+(2 n-1)^{\frac{1}{2}} g_{n-1}(t, x)+(2 n-3)^{\frac{1}{2}} g_{n-2}(t, x)+\ldots\right]
\end{aligned}
$$

系数动力学 将这些代入 20，我们得到

$$
\begin{aligned}
\frac{d}{d t} c_n(t)= & \int f(x)\left(\frac{\partial}{\partial t} g_n(t, x)\right) \omega(t, x) \mathrm{d} x+\int f(x) g_n(t, x)\left(\frac{\partial}{\partial t} \omega(t, x)\right) \mathrm{d} x \\
= & -t^{-1}(2 n+1)^{\frac{1}{2}}\left[n(2 n+1)^{-\frac{1}{2}} c_n(t)+(2 n-1)^{\frac{1}{2}} c_{n-1}(t)+(2 n-3)^{\frac{1}{2}} c_{n-2}(t)+\ldots\right] \\
& \quad-t^{-1} c_n(t)+t^{-1} f(t) g_n(t, t) \\
= & -t^{-1}(2 n+1)^{\frac{1}{2}}\left[(n+1)(2 n+1)^{-\frac{1}{2}} c_n(t)+(2 n-1)^{\frac{1}{2}} c_{n-1}(t)+(2 n-3)^{\frac{1}{2}} c_{n-2}(t)+\ldots\right] \\
& +t^{-1}(2 n+1)^{\frac{1}{2}} f(t)
\end{aligned}
$$

其中我们使用了 $g_n(t, t)=(2 n+1)^{\frac{1}{2}} P_n(1)=(2 n+1)^{\frac{1}{2}}$。将此向量化得到方程（3）：

$$
\begin{aligned}
\frac{d}{d t} c(t) & =-\frac{1}{t} A c(t)+\frac{1}{t} B f(t) \\
A_{n k} & = \begin{cases}(2 n+1)^{1 / 2}(2 k+1)^{1 / 2} & \text { if } n>k \\
n+1 & \text { if } n=k, \\
0 & \text { if } n<k\end{cases} \\
B_n & =(2 n+1)^{\frac{1}{2}}
\end{aligned}\tag{29}
$$

或者，我们可以将其写成

$$
\frac{d}{d t} c(t)=-t^{-1} D\left[M D^{-1} c(t)+1 f(t)\right]\tag{30}
$$

其中 $D:=\operatorname{diag}\left[(2 n+1)^{\frac{1}{2}}\right]_{n=0}^{N-1}, 1$ 是全 1 向量，状态矩阵 $M$ 是

$$
M=\left[\begin{array}{cccccc}
1 & 0 & 0 & 0 & \ldots & 0 \\
1 & 2 & 0 & 0 & \ldots & 0 \\
1 & 3 & 3 & 0 & \ldots & 0 \\
1 & 3 & 5 & 4 & \ldots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
1 & 3 & 5 & 7 & \ldots & N
\end{array}\right] \text {，即，} M_{n k}= \begin{cases}2 k+1 & \text {如果} k<n \\
k+1 & \text {如果} k=n \\
0 & \text {如果} k>n\end{cases}
$$

方程（29）是一个线性动态系统，除了被一个时间变化因子 $t^{-1}$ 扩展外，这是由缩放测度引起的。

重构 通过方程（19），在每个时刻 $t$，我们有
$$
\begin{aligned}
f(x) \approx g^{(t)}(x) & =\sum_n c_n(t) g_n(t, x) . \\
& =\sum_n c_n(t)(2 n+1)^{\frac{1}{2}} P_n\left(\frac{2 x}{t}-1\right) .
\end{aligned}
$$
#### D.4 傅里叶基的推导

在附录 D 的其余部分，我们考虑一些在 HiPPO 框架下可分析的附加基础。这些使用与傅里叶变换的各种形式相关的测度和基。

##### D.4.1 平移傅里叶

类似于 LMU，滑动傅里叶测度也具有固定的窗口长度 $\theta$ 参数，并将其沿着时间滑动。

测度 傅里叶基础 $e^{2 \pi i n x}$（对于 $n=0, \ldots, N-1$ ）可以看作是关于单位圆 $\{z:|z|=1\}$ 上均匀测度的正交多项式基础 $z^n$。通过变量变换 $z \rightarrow e^{2 \pi i x}$（从而将定义域从单位圆变换为 $[0,1]$ ），我们得到了常规的傅里叶基础 $e^{2 \pi i n x}$。复内积 $\langle f, g\rangle$ 定义为 $\int_0^1 f(x) \overline{g(x)} \mathrm{d} x$。注意基础 $e^{2 \pi i n x}$ 是正交的。
对于每个 $t$，我们将使用在 $[t-\theta, t]$ 上均匀的滑动测度，并将基础重新缩放为 $e^{2 \pi i n \frac{t-x}{v}}$（因此它们仍然是正交的，即，具有单位 1 的范数）：

$$
\begin{aligned}
\omega(t, x) & =\frac{1}{\theta} \mathbb{I}_{[t-\theta, t]} \\
p_n(t, x) & =e^{2 \pi i n \frac{t-x}{\theta}} .
\end{aligned}
$$

我们不使用倾斜（即，$\chi(t, x)=1$）。
导数

$$
\begin{aligned}
\frac{\partial}{\partial t} \omega(t, x) & =\frac{1}{\theta} \delta_t-\frac{1}{\theta} \delta_{t-\theta} \\
\frac{\partial}{\partial t} p_n(t, x) & =\frac{2 \pi i n}{\theta} e^{2 \pi i n \frac{t-x}{\theta}}=\frac{2 \pi i n}{\theta} p_n(t, x) .
\end{aligned}
$$

系数更新 将其代入方程（20）得到

$$
\begin{aligned}
\frac{d}{d t} c_n(t) & =\frac{2 \pi i n}{\theta} c_n(t)+\frac{1}{\theta} f(t) p_n(t, t)-\frac{1}{\theta} f(t-\theta) p_n(t, t-\theta) \\
& =\frac{2 \pi i n}{\theta} c_n(t)+\frac{1}{\theta} f(t)-\frac{1}{\theta} f(t-\theta) .
\end{aligned}
$$

注意到 $p_n(t, t)=p_n(t, t-\theta)=1$。此外，在时刻 $t$，我们不再访问 $f(t-\theta)$，但这隐含地表示在我们对函数的压缩表示中：$f=\sum_{k=0}^{N-1} c_k(t) p_k(t)$。因此，我们用 $\sum_{k=0}^{N-1} c_k(t)$ 近似 $f(t-\theta)$。最后，这导致

$$
\frac{d}{d t} c_n(t)=\frac{2 \pi i n}{\theta} c_n(t)+\frac{1}{\theta} f(t)-\frac{1}{\theta} \sum_{k=0}^{N-1} c_k(t) .
$$

因此 $\frac{d}{d t} c(t)=A c(t)+B f(t)$，其中

$$
A_{n k}=\left\{\begin{array}{ll}
-1 / \theta & \text {如果} k \neq n \\
(2 \pi i n-1) / \theta & \text {如果} k=n
\end{array}, \quad B_n=\frac{1}{\theta} .\right.
$$

重构 在每个时间步 $t$，我们有

$$
f(x) \approx \sum_n c_n(t) p_n(t, x)=\sum_n c_n(t) e^{2 \pi i \frac{t-z}\\

theta}}
$$

##### D.4.2 傅里叶循环单元

利用 HiPPO 框架，我们还可以推导出傅里叶循环单元（FRU）[79]。

测度 对于每个 $t$，我们将使用在 $[t-\theta, t]$ 上均匀的滑动测度和基础 $e^{2 \pi i n \frac{x}{\theta}}$：

$$
\begin{aligned}
\omega(t, x) & =\frac{1}{\theta} \mathbb{I}_{[t-\theta, t]} \\
p_n(t, x) & =e^{2 \pi i n \frac{z}{\theta}} .
\end{aligned}
$$

通常，基础相对于测度 $\omega(t, x)$ 不是正交的，但在 $t=\theta$ 处正交性成立。

导数
$$
\begin{aligned}
\frac{\partial}{\partial t} \omega(t, x) & =\frac{1}{\theta} \delta_t-\frac{1}{\theta} \delta_{t-\theta} \\
\frac{\partial}{\partial t} p_n(t, x) & =0 .
\end{aligned}
$$

系数更新 将其代入方程 20 得到

$$
\begin{aligned}
\frac{d}{d t} c_n(t) & =\frac{1}{\theta} f(t) p_n(t, t)-\frac{1}{\theta} f(t-\theta) p_n(t, t-\theta) \\
& =\frac{1}{\theta} e^{2 \pi i n \frac{t}{\theta}} f(t)-\frac{1}{\theta} e^{2 \pi i n \frac{t}{\theta}} f(t-\theta) .
\end{aligned}
$$

在时刻 $t$，我们不再访问 $f(t-\theta)$，但我们可以通过忽略此项来近似（假设函数 $f$ 仅在 $[0, \theta]$ 上定义，因此对于 $x<0$，$f(x)$ 可以设为零）。最后，这导致

$$
\frac{d}{d t} c_n(t)=\frac{e^{2 \pi i n \frac{t}{\theta}}}{\theta} f(t) .
$$

应用前向欧拉离散化（步长 $=1$），我们得到：

$$
c_n(k+1)=c_n(k)+\frac{e^{2 \pi i n \frac{t}{\theta}}}{\theta} f(t) .
$$

取实部得到傅里叶循环单元的更新 [79]。

请注意，每个 $n$ 中的递归是独立的，因此我们不需要选择 $n=0,1, \ldots, N-1$。因此，我们可以像 Zhang 等人 [79] 中那样选择随机频率 $n$​。

#### D.5 平移切比雪夫的推导

在 HiPPO 框架下，我们分析的最后一族正交多项式是切比雪夫多项式。切比雪夫多项式可以看作是傅里叶基的纯实模拟；例如，切比雪夫级数通过基变换与傅里叶余弦级数相关联 [8]。

测度和基础 基本的切比雪夫测度是 $\omega^{\text {cheb }}=\left(1-x^2\right)^{-1 / 2}$ 在 $(-1,1)$ 上。根据附录 B.1.3，我们选择以下测度和标准正交基多项式，用切比雪夫第一种多项式 $T_n$ 表示。

$$
\begin{aligned}
\omega(t, x) & =\frac{2}{\theta \pi} \omega^{\text {cheb }}\left(\frac{2(x-t)}{\theta}+1\right) \mathbb{I}_{(t-\theta, t)} \\
& =\frac{1}{\theta \pi}\left(\frac{x-t}{\theta}+1\right)^{-1 / 2}\left(-\frac{x-t}{\theta}\right)^{-1 / 2} \mathbb{I}_{(t-\theta, t)} \\
p_n(t, x) & =\sqrt{2} T_n\left(\frac{2(x-t)}{\theta}+1\right) \quad \text { for } n \geq 1, \\
p_0(t, x) & =T_0\left(\frac{2(x-t)}{\theta}+1\right) .
\end{aligned}
$$

注意，在端点处，这些计算结果为

$$
\begin{aligned}
p_n(t, t) & = \begin{cases}\sqrt{2} T_n(1)=\sqrt{2} & n \geq 1 \\
T_n(1)=1 & n=0\end{cases} \\
p_n(t, t-\theta) & = \begin{cases}\sqrt{2} T_n(-1)=\sqrt{2}(-1)^n & n \geq 1 \\
T_n(-1)=1 & n=0\end{cases}
\end{aligned}
$$

倾斜测度 现在我们选择

$$
\chi^{(t)}=8^{-1 / 2} \theta \pi \omega^{(t)},
$$

所以

$$
\frac{\omega}{\chi^2}=\frac{1}{\frac{\theta^2 \pi^2}{8} \omega}=\frac{8}{\theta \pi}\left(\frac{x-t}{\theta}+1\right)^{1 / 2}\left(-\frac{x-t}{\theta}\right)^{1 / 2} \mathbb{I}_{(t-\theta, t)}
$$

其积分为 1。

我们还选择 $\lambda_n=1$​ 作为规范正交基，因此

$$
g^{(t)}=p_n^{(t)} \chi^{(t)}
$$

导数 密度的导数是

$$
\frac{\partial}{\partial t} \frac{\omega}{\chi}=\frac{\partial}{\partial t} \frac{8^{1 / 2}}{\theta \pi} \mathbb{I}_{(t-\theta, t)}=\frac{8^{1 / 2}}{\theta \pi}\left(\delta_t-\delta_{t-\theta}\right) .
$$

我们分别考虑对 $n=0, n$ 为偶数和奇数的多项式进行微分，使用方程 (TI)。为了方便起见，定义 $z=\frac{2(x-t)}{\theta}+1$。首先，对于 $n$ 为偶数，

$$
\begin{aligned}
\frac{\partial}{\partial t} p_n(t, x) & =-\frac{2^{\frac{3}{2}}}{\theta} T_n^{\prime}\left(\frac{2(x-t)}{\theta}+1\right) \\
& =-\frac{2^{\frac{3}{2}}}{\theta} T_n^{\prime}(z) \\
& =-\frac{2^{\frac{3}{2}}}{\theta} \cdot 2 n\left(T_{n-1}(z)+T_{n-3}(z)+\cdots+T_1(z)\right) \\
& =-\frac{4 n}{\theta}\left(p_{n-1}(t, x)+p_{n-3}(t, x)+\cdots+p_1(t, x)\right)
\end{aligned}
$$

对于奇数 $n$，

$$
\begin{align*}
\frac{\partial}{\partial t} p_n(t, x) & =-\frac{2^{\frac{3}{2}}}{\theta} T_n^{\prime}\left(\frac{2(x-t)}{\theta}+1\right) \\
& =-\frac{2^{\frac{3}{2}}}{\theta} T_n^{\prime}(z) \\
& =-\frac{2^{\frac{3}{2}}}{\theta} \cdot 2 n\left(T_{n-1}(z)+T_{n-3}(z)+\cdots+T_1(z)+\frac{1}{2} T_0(z)\right) \\
& =-\frac{4 n}{\theta}\left(p_{n-1}(t, x)+p_{n-3}(t, x)+\cdots+2^{-\frac{1}{2}} p_0(t, x)\right)
\end{align*}
$$

和

$$
\frac{\partial}{\partial t} p_0(t, x)=0 .
$$

系数动态

$$
\begin{aligned}
c_n(t) & =\int f(x) p_n(t, x) \frac{2^{3 / 2}}{\theta \pi} \mathbb{I}_{(t-\theta, t)} \mathrm{d} x \\
\frac{d}{d t} c_n(t) & =\int f(x) \frac{\partial}{\partial t} p_n(t, x) \frac{2^{3 / 2}}{\theta \pi} \mathbb{I}_{(t-\theta, t)} \mathrm{d} x+\frac{2^{3 / 2}}{\theta \pi} f(t) p_n(t, t)-\frac{2^{3 / 2}}{\theta \pi} f(t-\theta) p_n(t, t-\theta) \\
& =-\frac{4 n}{\theta}\left(c_{n-1}+c_{n-3}+\ldots\right)+\frac{2^{3 / 2}}{\theta \pi} f(t)\left\{\begin{array}{ll}
\sqrt{2} & n \geq 1 \\
1 & n=0
\end{array},\right.
\end{aligned}
$$

其中我们取 $f(t-\theta)=0$，因为我们不再能够访问它（这也适用于 $t<\theta$）。通常情况下，我们可以将这写成线性动态方程

$$
\begin{aligned}
\frac{d}{d t} c(t) & =-\frac{1}{\theta} A c(t)+\frac{1}{\theta} B f(t) \\
A & =4\left[\begin{array}{ccccc}
0 & & & \ldots \\
2^{-\frac{1}{2}} & 0 & & & \\
0 & 2 & 0 & & \ldots \\
2^{-\frac{1}{2}} \cdot 3 & 0 & 3 & 0 & \\
& \ddots & & \ddots &
\end{array}\right] \\
B & =\frac{2^{3 / 2}}{\pi}\left[\begin{array}{c}
1 \\
\sqrt{2} \\
\sqrt{2} \\
\sqrt{2} \\
\vdots
\end{array}\right]
\end{aligned}
$$

重构 在区间 $(t-\theta, t)$ 中，

$$
f(x) \approx \sum_{n=0}^{N-1} c_n(t) p_n(t, x) \chi(t, x) .
$$
