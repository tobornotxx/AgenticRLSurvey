# Agent Loop管理与Agentic RL 文献调研

> 调研时间：2026年4月24日
> 主题：DeepResearch超长上下文管理方案、上下文折叠、记忆融合

---

## 目录

1. [ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization](#1-resum)
2. [SUPO: Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management](#2-supo)
3. [IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling](#3-iterresearch)
4. [Context-Folding: Scaling Long-Horizon LLM Agent via Context-Folding](#4-context-folding)
5. [AgentFold: Long-Horizon Web Agents with Proactive Context Management](#5-agentfold)
6. [MemOS: A Memory OS for AI System](#6-memos)
7. [MIRIX: Multi-Agent Memory System for LLM-Based Agents](#7-mirix)
8. [EverMemOS (EverOS): A Self-Organizing Memory Operating System](#8-evermemos)
9. [Mem-α: Learning Memory Construction via Reinforcement Learning](#9-mem-alpha)

---

## 1. ReSum

**论文标题**：ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization

**作者**：Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang 等 (Tongyi Lab, Alibaba)

**链接**：[arXiv:2509.13313](https://arxiv.org/abs/2509.13313)

**发表**：ICML 2026 投稿（v3, 2026年3月）

### 1.1 研究问题

LLM Web Agent在处理知识密集型长程任务时，面临**探索需求与上下文窗口限制之间的根本矛盾**。以ReAct范式为代表的Agent将所有Thought、Action、Observation追加到历史中，在需要大量工具交互的复杂查询中很快耗尽上下文预算。现有解决方案（如MEM1、MemAgent）依赖**架构修改**（如生成内部记忆token），破坏了与已有Agent的兼容性，并且需要昂贵的端到端重训练。

### 1.2 基本假设

1. 通过**周期性摘要压缩**交互历史，可以在不修改Agent架构的前提下实现无限探索
2. 摘要工具可以作为即插即用的外部组件，代表一种**范式增强**（paradigmatic enhancement）而非架构变更
3. 标准Agent虽然可以从摘要中推理，但并未针对压缩上下文进行优化，通过RL训练可以弥补这一差距

### 1.3 遇到的困难

1. **上下文预算耗尽**：在BrowseComp等困难基准上，失败的轨迹频繁超过32K token限制
2. **摘要质量问题**：通用LLM在从冗长、嘈杂的交互历史中提取关键证据方面能力不足
3. **信用分配困难**：在长程任务中，轨迹被摘要事件自然分割成多个segment，传统RL难以将最终奖励正确传播到所有segment
4. **范式适配问题**：标准Agent未经训练处理 $q' = (q, s)$ 形式的摘要条件输入

### 1.4 算法解决方案

#### 1.4.1 ReSum范式

**核心设计**：周期性调用外部摘要工具压缩交互历史

**算法流程**：

1. **轨迹初始化**：从用户查询 $q$ 开始，$\mathcal{H}_0 = (q)$
2. **迭代交互**：在第 $t$ 轮，生成推理步骤和工具调用：

$$(\tau_t, a_t) \sim \pi_\theta(\cdot | \mathcal{H}_{t-1})$$

3. **历史更新**：

$$\mathcal{H}_t = \mathcal{H}_{t-1} \circ (\tau_t, a_t, o_t)$$

4. **上下文压缩**：当触发条件激活时（如接近上下文限制），调用摘要工具：

$$s \sim \pi_{\text{sum}}(\cdot | \mathcal{H}_t)$$

然后形成压缩状态 $q' = (q, s)$ 并重置历史：$\mathcal{H}_t \leftarrow (q')$

5. **轨迹终止**：Agent积累足够信息后在 `<answer>` 标签中输出最终答案

#### 1.4.2 ReSumTool-30B

通过三阶段流水线开发专用摘要模型：

1. **教师模型选择**：选择GPT-OSS-120B作为教师模型
2. **数据合成**：在SailorFog-QA基准上收集9000+高质量 $\langle \text{Conversation}, \text{Summary} \rangle$ 对
3. **蒸馏训练**：基于Qwen3-30B-A3B-Thinking进行SFT，batch_size=64，2 epochs，learning rate $7 \times 10^{-6}$

#### 1.4.3 ReSum-GRPO（核心算法）

**动机**：标准GRPO无法处理被摘要事件分割的长轨迹中的信用分配问题

**轨迹分割**：完整ReSum轨迹经历 $K$ 次摘要事件后，被自然划分为 $K+1$ 个segment：

$$\mathcal{H}^{(1)} = (q^{(0)}, \tau_1, a_1, o_1, \ldots, \tau_{t_1}, a_{t_1}, o_{t_1})$$
$$\mathcal{H}^{(2)} = (q^{(1)}, \tau_{t_1+1}, a_{t_1+1}, o_{t_1+1}, \ldots, \tau_{t_2}, a_{t_2}, o_{t_2})$$
$$\vdots$$
$$\mathcal{H}^{(K+1)} = (q^{(K)}, \tau_{t_K+1}, a_{t_K+1}, o_{t_K+1}, \ldots, \tau_T, a_T)$$

其中 $q^{(0)} = q$ 是初始查询，$q^{(k)} = (q, s^{(k)})$ 是第 $k$ 次摘要后的压缩状态。

**奖励计算**：使用统一的轨迹级奖励信号，基于LLM-as-Judge策略：

$$R(a^*, a_T) \in \{0, 1\}$$

**优势广播（Advantage Broadcasting）**：

对轨迹 $g$，提取最终答案 $a_{g,T}$ 并计算轨迹级奖励 $R_g \in \{0,1\}$，在组内归一化：

$$\hat{A}_g = \frac{R_g - \text{mean}(\{R_1, \ldots, R_G\})}{\text{std}(\{R_1, \ldots, R_G\})}$$

将该优势广播到rollout $g$ 内所有segment：$\hat{A}_g^{(i)} = \hat{A}_g, \forall i \in \{1, \ldots, n_g\}$

**GRPO目标函数**：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{g=1}^{G} n_g} \sum_{g=1}^{G} \sum_{i=1}^{n_g} \min\left(r_g^{(i)}(\theta) \hat{A}_g^{(i)}, \text{clip}(r_g^{(i)}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) \hat{A}_g^{(i)}\right)\right]$$

### 1.5 实验安排与结果

**基准**：GAIA、BrowseComp、BrowseComp-zh

**评估**：使用Qwen2.5-72B-Instruct作为打分模型，报告Pass@1和Pass@3

**主要结果**：

| 设置 | 方法 | BrowseComp Pass@1 |
|------|------|-------------------|
| 训练-free | ReAct (WebSailor-30B) | 12.8% |
| 训练-free | ReSum + ReSumTool-30B | 16.0% |
| RL训练 | GRPO (ReAct) | 14.3% |
| RL训练 | ReSum-GRPO | 18.3% |
| RL训练 | MEM1-GRPO | 19.5% |

**关键发现**：
1. ReSum在训练-free设置下平均比ReAct提升4.5%
2. ReSum-GRPO在仅1K训练样本下进一步提升8.2%
3. 30B Agent通过ReSum可超越Claude-4-Sonnet (12.2%)和Kimi-K2 (14.1%)等商用模型
4. ReSum训练开销约为GRPO的1.5倍，但推理效率远优于MEM1（token消耗约为MEM1的1/3）

### 1.6 欠缺与不足

1. **摘要触发机制固定**：采用规则触发（接近上下文限制时触发），未实现Agent自主决定何时摘要
2. **信息损失风险**：摘要过程不可避免地丢失细节，缺乏质量控制机制来保证长程依赖的保留
3. **外部摘要工具依赖**：需要额外部署一个30B摘要模型，增加了系统复杂度
4. **优势广播的粗糙性**：所有segment共享同一优势值，无法区分不同segment的贡献差异

---

## 2. SUPO

**论文标题**：Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management

**作者**：Miao Lu, Weiwei Sun, Weihua Du, Zhan Ling, Xuesong Yao, Kang Liu, Jiecao Chen (Google)

**链接**：[arXiv:2510.06727](https://arxiv.org/abs/2510.06727)

> ⚠️ **注意**：原知乎链接 https://zhuanlan.zhihu.com/p/1959726205738681596 返回403无法访问。论文HTML版本也未能获取。以下内容基于arXiv摘要和Context-Folding同组的关联工作整理。

### 2.1 研究问题

研究**LLM Agent在长程多轮工具使用中的RL微调**问题，上下文长度成为根本瓶颈。现有RL流水线面临：指令遵循退化、rollout成本过高，以及严格的上下文限制。

### 2.2 基本假设

1. 通过在训练中引入**基于摘要的上下文管理**，可以周期性压缩工具使用历史，保留任务相关信息
2. 可以推导出一种**策略梯度表示**，使标准LLM RL基础设施能够以端到端方式同时优化工具使用行为和摘要策略
3. 摘要增强的策略优化可以使Agent突破固定上下文限制进行训练

### 2.3 遇到的困难

1. **上下文长度瓶颈**：多轮工具使用中上下文快速增长，超出固定窗口限制
2. **RL训练效率**：长轨迹的rollout成本极高
3. **摘要与工具使用的联合优化**：需要同时学习何时/如何摘要以及如何使用工具

### 2.4 算法解决方案

**SUPO (SUmmarization augmented Policy Optimization)**：

**核心创新**：推导策略梯度表示，使得摘要策略和工具使用策略可以在标准LLM RL框架下端到端优化。

**算法步骤**：
1. Agent在多轮交互中使用工具
2. 周期性地，LLM生成摘要压缩历史上下文
3. 通过推导的策略梯度，同时优化：
   - 工具调用决策
   - 摘要内容和时机
4. 支持在**测试时扩展**：测试时的最大摘要轮数可以超过训练时的设置

### 2.5 实验安排与结果

在交互式函数调用和搜索任务上进行实验：
- SUPO显著提升成功率
- 保持相同或更低的工作上下文长度
- 在复杂搜索任务上，测试时扩展摘要轮数可进一步提升性能

### 2.6 欠缺与不足

1. **论文全文难以获取**：HTML版本不可用，算法细节信息有限
2. **与Context-Folding的关系**：来自同一研究组（Google），SUPO是Context-Folding框架FoldGRPO的RL实例化，但具体差异和改进点不清楚
3. 端到端优化的计算开销可能较高

---

## 3. IterResearch

**论文标题**：IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling

**作者**：Guoxin Chen, Zile Qiao, Xuanzhong Chen 等 (人民大学、Tongyi Lab, Alibaba)

**链接**：[arXiv:2511.07327](https://arxiv.org/abs/2511.07327)

**发表**：ICLR 2026 Camera Ready

### 3.1 研究问题

现有深度研究Agent依赖**单上下文范式（mono-contextual paradigm）**，将所有信息累积在单一、不断膨胀的上下文窗口中。这导致两个核心问题：
1. **上下文窒息（context suffocation）**：上下文窗口填满后，模型推理空间逐渐缩小
2. **噪声污染（noise contamination）**：不相关信息和早期探索错误永久嵌入上下文

### 3.2 基本假设

1. 有效的长程研究需要**周期性综合**和**战略性遗忘**——这是当前单上下文方法所缺失的能力
2. 通过MDP启发的架构，每个状态是战略性重构的工作空间，而非不断增长的历史
3. 未来探索仅依赖当前重构状态而非整体历史，可以在任意探索深度上保持一致的推理能力

### 3.3 遇到的困难

1. **状态空间爆炸**：单上下文方法 $|s_t| \propto t$，线性增长不可持续
2. **效率激励缺失**：二元终端奖励同等对待所有成功轨迹，无论计算成本
3. **分布式训练不稳定**：迭代范式将轨迹分解为多个独立训练样本，样本数在不同问题间差异很大

### 3.4 算法解决方案

#### 3.4.1 MDP启发的迭代研究范式

定义元组 $\langle \mathcal{S}, \mathcal{D}, \mathcal{E}, \mathcal{T}, R \rangle$：

- **状态空间** $\mathcal{S}$：$s_t = (q, \mathcal{M}_t, \{a_{t-1}, \text{TR}_{t-1}\})$，包含问题 $q$、**演化报告** $\mathcal{M}_t$（压缩记忆）、上一步交互
- **决策空间** $\mathcal{D}$：

$$d_t = [\underbrace{\text{Think}_t, \mathcal{M}_{t+1}}_{\text{Internal Thought}}, \underbrace{a_t}_{\text{External Action}}] \sim \pi(\cdot | s_t)$$

- **状态转移函数** $\mathcal{T}$（关键创新——工作空间重构）：

$$s_{t+1} = \mathcal{T}(s_t, d_t, \text{TR}_t) = (q, \mathcal{M}_{t+1}, \{a_t, \text{TR}_t\})$$

**与单上下文的对比**：

$$\underbrace{s_t^{\text{mono}} = [q, a_0, \text{TR}_0, \ldots, a_{t-1}, \text{TR}_{t-1}]}_{\text{Mono-contextual: } \mathcal{O}(t) \text{ growth}} \quad \text{vs.} \quad \underbrace{s_t^{\text{iter}} = (q, \mathcal{M}_t, \{a_{t-1}, \text{TR}_{t-1}\})}_{\text{IterResearch: } \mathcal{O}(1) \text{ constant}}$$

#### 3.4.2 EAPO (Efficiency-Aware Policy Optimization)

**折扣奖励塑形**：

$$r_t = \gamma^{T-t} \cdot R_T, \quad \gamma \in (0,1)$$

其中 $T$ 是终止步，$\gamma = 0.995$。这创造了隐式效率压力：更早完成任务的步骤获得更高奖励。

**示例**：对于步骤 $t=3$，5步轨迹获得 $r_3^A = \gamma^2 \approx 0.99$，20步轨迹获得 $r_3^B = \gamma^{17} \approx 0.918$，7.8%的奖励差异引导策略走向更高效的探索。

**自适应下采样**：

$$|\mathcal{C}_{\text{train}}| = \lfloor \frac{|\mathcal{C}|}{\text{DP}_{\text{size}}} \rfloor \times \text{DP}_{\text{size}}$$

确保数据损失<1%同时保持分布式训练稳定。

**策略优化目标**：

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim \mathcal{Q}, \mathcal{C}_{\text{train}} \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{|\mathcal{C}_{\text{train}}|} \sum_{i=1}^{G} \sum_{t=1}^{T_i} \min\left(\rho_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(\rho_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t}\right) \right]$$

其中所有 $\sum_{i=1}^{G} T_i$ 轮构成**一个组**，优势在组内归一化：$\hat{A}_{i,t} = \frac{r_{i,t} - \mu_r}{\sigma_r}$

### 3.5 实验安排与结果

**基准**（6个）：HLE、BrowseComp、BrowseComp-zh、GAIA、Xbench-DeepSearch、SEAL-0

**主要结果**：

| 模型 | HLE | BC | BC-zh | GAIA | Xbench-DS | SEAL-0 |
|------|-----|----|----|------|-----------|--------|
| WebSailor-32B | 9.6 | 10.5 | 25.5 | 53.2 | 53.3 | 16.2 |
| MiroThinker-32B | 19.1 | 17.2 | 29.4 | 64.1 | 56.0 | - |
| **IterResearch-30B-A3B** | **28.8** | **37.3** | **45.2** | **72.8** | **71.0** | **39.6** |
| OpenAI DeepResearch | 26.6 | 51.5 | 42.9 | 67.4 | - | - |

**交互缩放**：在BrowseComp上从2轮扩展到2048轮，准确率从3.5%提升至42.5%（仅用40K上下文）

**跨范式知识迁移**：IterResearch生成的轨迹可提升单上下文Agent性能（+5.4pp）

**作为Prompt策略**：无需训练直接应用于o3和DeepSeek-V3.1，在BrowseComp上分别+12.7pp和+19.2pp

### 3.6 欠缺与不足

1. **报告质量依赖**：演化报告的压缩质量直接决定后续推理质量，报告中的信息丢失不可逆
2. **训练数据需求**：SFT阶段需要110K合成轨迹，数据构建成本较高
3. **推理开销**：BrowseComp上平均生成376K token（不含工具响应），推理成本极高
4. **单一报告结构**：所有信息压缩到单一报告中，可能不适合需要多维度记忆的复杂任务

---

## 4. Context-Folding

**论文标题**：Scaling Long-Horizon LLM Agent via Context-Folding

**作者**：Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, Jiecao Chen (Google)

**链接**：[arXiv:2510.11967](https://arxiv.org/abs/2510.11967)

> ⚠️ **注意**：论文HTML版本解析失败，以下内容基于arXiv摘要和相关工作中的引用整理。

### 4.1 研究问题

LLM Agent在长程任务上受到**上下文长度的根本限制**。需要一种框架使Agent能够**主动管理其工作上下文**。

### 4.2 基本假设

1. Agent可以程序化地**分支进入子轨迹**处理子任务，完成后**折叠**（fold）子轨迹
2. 折叠操作将中间步骤压缩为简洁摘要，同时保留关键结果
3. 通过端到端RL训练（FoldGRPO），Agent可以学习有效的任务分解和上下文管理

### 4.3 遇到的困难

1. 基于摘要的上下文管理（如全历史摘要）可能导致关键细节不可逆丢失
2. 需要同时学习：何时分支、如何执行子任务、如何折叠
3. RL训练需要特定的过程奖励来鼓励有效的任务分解

### 4.4 算法解决方案

**Context-Folding框架**：

Agent可以：
1. **分支（Branch）**：进入子轨迹处理子任务
2. **折叠（Fold）**：完成子任务后折叠子轨迹，将中间步骤压缩为简洁的结果摘要

**FoldGRPO**：端到端RL框架，设计了**特定的过程奖励**来鼓励有效的任务分解和上下文管理。

### 4.5 实验安排与结果

在Deep Research和SWE任务上评估：
- 折叠Agent**匹配或超越ReAct基线**
- 使用的**活跃上下文小10倍**
- **显著优于**基于摘要的上下文管理方法

### 4.6 欠缺与不足

1. **论文细节不足**：HTML版本不可用，算法具体公式和步骤无法完整呈现
2. 分支-折叠操作增加了Agent行为空间的复杂度
3. 过程奖励的设计可能需要针对不同任务进行调整

---

## 5. AgentFold

**论文标题**：AgentFold: Long-Horizon Web Agents with Proactive Context Management

**作者**：Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin 等 (Tongyi Lab, Alibaba)

**链接**：[arXiv:2510.24699](https://arxiv.org/abs/2510.24699)

### 5.1 研究问题

LLM Web Agent在长程任务中面临上下文管理的**根本权衡**：
1. ReAct范式的Agent累积嘈杂的原始历史，导致**上下文饱和**
2. 固定摘要全部历史的方法面临**关键细节不可逆丢失**的风险

### 5.2 基本假设

1. 理想的Agent应将其内部上下文视为**动态认知工作空间**来主动管理，而非被动填充的日志
2. 灵感来自人类认知中的**回溯巩固（retrospective consolidation）**过程
3. Agent可以在**多尺度**上管理历史轨迹：精细凝缩保留关键细节，深度巩固抽象化整个子任务

### 5.3 遇到的困难

1. **信息保留的累积风险**：如果全历史每次重摘要有1%的关键细节丢失概率，步骤1的信息存活到步骤100只有 $0.99^{100} \approx 36.6\%$，到步骤500仅 $0.99^{500} \approx 0.66\%$
2. 需要训练数据展示复杂的行动与上下文管理的**交互配合**
3. 即使最先进的LLM也无法通过prompt engineering可靠产生AgentFold的结构化多部分响应

### 5.4 算法解决方案

#### 5.4.1 AgentFold上下文设计

上下文 $C_t$ 在步骤 $t$ 时为四元组：

$$C_t = (Q, T, S_{t-2}, I_{t-1})$$

- $Q$：不变的用户问题
- $T$：可用工具列表
- $S_{t-2}$：**多尺度状态摘要（Multi-Scale State Summaries）**——长期记忆
- $I_{t-1}$：**最新交互（Latest Interaction）**——即时工作记忆

多尺度状态摘要表示为有序序列：

$$S_t = (s_{x_1,y_1}, s_{x_2,y_2}, \ldots, s_{x_m,y_m})$$

其中 $s_{x,y}$ 是从步骤 $x$ 到 $y$ 的文本摘要。单步摘要为 $s_{x,x}$（$y=x$），多步摘要为 $s_{x,y}$（$y > x$）。

#### 5.4.2 AgentFold响应结构

每步生成四元组响应：

$$R_t = \text{AgentFold}(C_t; \theta) \rightarrow (th_t, f_t, e_t, a_t)$$

- $th_t$：**思考过程**（chain-of-thought）
- $f_t$：**折叠指令**（JSON格式）：$f_t = \{``\text{range}": [k, t-1], ``\text{summary}": ``\sigma_t"\}$
- $e_t$：**解释**
- $a_t$：**动作**（工具调用或最终答案）

**折叠指令的两种模式**：

1. **精细凝缩（Granular Condensation）**（$k = t-1$）：仅折叠最新交互为新的细粒度摘要
2. **深度巩固（Deep Consolidation）**（$k < t-1$）：融合最新交互与一系列先前摘要为单一粗粒度摘要

#### 5.4.3 训练方法（Fold-Generator + SFT）

1. 使用强大的开源LLM构建Fold-Generator数据收集流水线
2. 通过**拒绝采样**机制过滤不符合格式的步骤
3. 收集高质量 $\{(C_t, R_t^*)\}_N$ 对
4. 基于Qwen3-30B-A3B-Instruct-2507进行**监督微调（SFT）**

### 5.5 实验安排与结果

**基准**：BrowseComp、BrowseComp-ZH、WideSearch、GAIA

| Agent | BrowseComp | BC-ZH | WideSearch | GAIA |
|-------|-----------|-------|------------|------|
| WebSailor-72B | 12.0 | 30.1 | - | 55.4 |
| DeepSeek-V3.1-671B | 30.0 | 49.2 | - | 63.1 |
| OpenAI o4-mini | 28.3 | 44.3 | - | - |
| **AgentFold-30B-A3B** | **36.2** | **47.3** | **62.1** | **67.0** |

**上下文效率**：100步后上下文仅约7K token（模型上限128K），ReAct约91K token，**压缩率92%**

**交互缩放**：可扩展到256+步，30B模型持续超越355B GLM-4.5基线

### 5.6 欠缺与不足

1. **仅使用SFT训练**：未利用RL来发现最优折叠策略（作者明确指出这是下一步工作）
2. **数据收集成本**：Fold-Generator需要强大LLM生成、拒绝采样，数据构建成本高
3. **折叠决策的最优性**：SFT学到的折叠策略可能不是最优的
4. **多尺度摘要的冗余**：随步数增长，摘要块数仍线性增长（虽然增速远低于ReAct）

---

## 6. MemOS

**论文标题**：MemOS: A Memory OS for AI System

**作者**：Zhiyu Li, Shichao Song, Chenyang Xi 等 (MemTensor, 上海交大, 同济大学 等)

**链接**：[arXiv:2507.03724](https://arxiv.org/abs/2507.03724)

### 6.1 研究问题

LLM缺乏**良好定义的记忆管理系统**，阻碍了长上下文推理、持续个性化和知识一致性的发展。现有模型主要依赖静态参数和短期上下文状态，无法追踪用户偏好或长期更新知识。RAG虽然引入了外部知识，但仍然是无状态的临时方案，缺乏生命周期控制。

### 6.2 基本假设

1. 记忆应被视为**可管理的系统资源**，类似操作系统管理CPU、存储和I/O
2. 需要统一表示、调度和演化**纯文本记忆、激活记忆和参数记忆**三种异构记忆类型
3. 记忆管理将从人工定义转向**模型自主定义**

### 6.3 遇到的困难

1. **记忆异构性**：参数记忆、激活记忆、纯文本记忆的来源、生命周期和调度方式差异巨大
2. **跨平台记忆孤岛**：用户记忆被锁定在特定平台内
3. **安全与隐私**：多用户、多Agent环境下的访问控制和合规性

### 6.4 系统架构

#### 6.4.1 三种记忆类型

1. **纯文本记忆（Plaintext Memory）**：外部知识片段（检索段落、知识图谱节点、prompt模板），可编辑、可追踪
2. **激活记忆（Activation Memory）**：推理时生成的中间状态（KV-cache、隐藏状态、注意力权重），短期、动态
3. **参数记忆（Parameter Memory）**：模型权重中编码的知识（$W_{\text{MLP}}^l$, $W_K^l$, $W_V^l$），长期、隐式

**记忆转换路径**：
- 纯文本 $\Rightarrow$ 激活：高频纯文本转为KV缓存
- 纯文本/激活 $\Rightarrow$ 参数：稳定知识蒸馏为LoRA模块
- 参数 $\Rightarrow$ 纯文本：冷门参数卸载为外部存储

#### 6.4.2 MemCube（核心资源单元）

每个MemCube包含：
- **记忆载荷（Memory Payload）**：语义内容
- **元数据（Metadata）**：
  - **描述标识符**：时间戳、来源签名、语义类型
  - **治理属性**：访问控制、生存周期策略（TTL）、优先级
  - **行为使用指标**：访问频率、上下文指纹、版本链

#### 6.4.3 三层架构

1. **接口层**：MemReader（语义解析）、Memory API（统一接口）、Memory Pipeline（组合操作）
2. **操作层**：MemOperator（记忆组织与检索）、MemScheduler（类型感知调度）、MemLifecycle（生命周期管理：Generated → Activated → Merged → Archived → Expired）
3. **基础设施层**：MemGovernance（权限控制）、MemVault（存储路由）、MemLoader/MemDumper（迁移同步）、MemStore（发布订阅）

### 6.5 实验安排与结果

在LOCOMO基准上评估：

| 方法 | 整体LLM-Judge | 单跳 | 多跳 | 开放域 | 时序推理 |
|------|-------------|------|------|--------|---------|
| LangMem | 55.76 | 68.21 | 56.74 | 49.65 | 24.09 |
| Zep | 41.62 | 50.42 | 42.20 | 38.19 | 19.11 |
| OpenAI-Memory | 52.75 | 61.83 | 60.28 | 32.99 | 28.25 |
| Mem0 | 64.57 | 73.33 | 58.75 | 45.83 | 52.34 |
| **MemOS-0630** | **73.31** | **78.44** | **64.30** | **55.21** | **73.21** |

KV记忆加速：在Qwen2.5-72B上长上下文短查询场景下TTFT降低91.4%

### 6.6 欠缺与不足

1. **系统复杂度高**：三层架构加多种记忆类型，部署和维护成本高
2. **缺乏RL优化**：记忆管理策略主要依赖工程设计，未利用RL进行自动化优化
3. **评估范围有限**：主要在对话记忆基准（LOCOMO）上评估，缺乏Agent工具使用场景的验证
4. **跨模型兼容性未验证**：系统设计面向通用，但实际跨模型迁移效果未知
5. **论文偏重系统设计**：更像系统论文而非算法论文，缺乏严格的形式化定义和理论分析

---

## 7. MIRIX

**论文标题**：MIRIX: Multi-Agent Memory System for LLM-Based Agents

**作者**：Yu Wang, Xi Chen

**链接**：[arXiv:2507.07957](https://arxiv.org/abs/2507.07957)

### 7.1 研究问题

现有AI Agent的记忆解决方案**依赖扁平、窄范围的记忆组件**，限制了个性化、抽象化和长期可靠回忆用户特定信息的能力。

### 7.2 基本假设

1. Agent记忆需要**多种结构化类型**来处理不同信息维度
2. **多Agent框架**可以动态控制和协调记忆的更新与检索
3. 记忆系统应超越文本，拥抱**多模态体验**

### 7.3 遇到的困难

1. 处理海量多模态数据（如20000张高分辨率截图）的存储效率问题
2. 记忆类型间的协调和冲突解决
3. 大规模长期用户数据的准确检索

### 7.4 算法解决方案

**六种记忆类型**：
1. **Core Memory**：核心用户信息
2. **Episodic Memory**：事件记忆（时间戳标记）
3. **Semantic Memory**：事实知识和声明性信息
4. **Procedural Memory**：程序性知识
5. **Resource Memory**：资源记忆
6. **Knowledge Vault**：知识库

**多Agent框架**：专门的Agent动态控制各类记忆的更新和检索

### 7.5 实验安排与结果

1. **ScreenshotVQA**：处理近20000张高分辨率截图序列
   - 比RAG基线**准确率高35%**
   - **存储需求降低99.9%**
2. **LOCOMO**：长对话基准
   - 达到**85.4% SOTA性能**

### 7.6 欠缺与不足

1. **依赖指令遵循能力**：小模型（如GPT-4.1-mini以下）难以正确使用复杂的多类型记忆工具
2. **缺乏训练优化**：预定义指令和工具集，未进行RL训练（Mem-α论文指出即使GPT-4o也难以有效利用MIRIX）
3. **系统复杂度**：六种记忆类型加多Agent协调，工程实现复杂
4. **评估场景有限**：仅在截图VQA和对话两个场景验证

---

## 8. EverMemOS

**项目名称**：EverOS (原EverMemOS)

**机构**：EverMind AI

**链接**：[GitHub](https://github.com/EverMind-AI/EverOS)

**关联论文**：
- EverMemOS: A Self-Organizing Memory Operating System (arXiv:2601.02163, 2026)
- HyperMem: Hypergraph Memory for Long-Term Conversations (arXiv:2604.08256, 2026)

> ⚠️ **注意**：这是一个开源项目而非单篇论文。以下基于GitHub README和关联论文信息整理。

### 8.1 研究问题

如何为自我演化Agent构建、评估和集成**长期记忆**系统。

### 8.2 项目结构

```
EverOS/
├── methods/
│   ├── EverCore/      # 长期记忆操作系统
│   └── HyperMem/      # 超图记忆架构
├── benchmarks/
│   ├── EverMemBench/   # 记忆质量评估
│   └── EvoAgentBench/  # Agent自我演化评估
└── use-cases/          # 使用案例
```

### 8.3 核心方法

1. **EverCore**：受生物印记启发的自组织记忆操作系统。从对话中提取、结构化和检索长期知识。
2. **HyperMem**：基于超图的分层记忆架构。通过超边捕获高阶关联，将记忆组织为话题、事件和事实三层，实现粗到细的长期对话检索。在LoCoMo基准达到92.73%。

### 8.4 评估框架

1. **EverMemBench**：三层记忆质量评估——事实回忆、应用推理、个性化泛化
2. **EvoAgentBench**：Agent自我演化评估——测量迁移效率、错误避免和技能命中质量

### 8.5 实验结果

- EverCore在LoCoMo基准达到93%整体准确率
- HyperMem在LoCoMo达到92.73%

### 8.6 欠缺与不足

1. **侧重工程实现**：更偏向工程框架而非算法创新
2. **缺乏Agent工具使用场景**：主要面向对话记忆，未涉及DeepResearch等Agent场景
3. **论文尚未充分公开**：关联论文（2026年）详细内容有限
4. **与RL的结合不明确**：未明确提出RL训练记忆管理的方案

---

## 9. Mem-α

**论文标题**：Mem-α: Learning Memory Construction via Reinforcement Learning

**作者**：Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, Xiaojian Wu (Anuttacon, UCSD, Stanford)

**链接**：[arXiv:2509.25911](https://arxiv.org/abs/2509.25911)

### 9.1 研究问题

当前记忆增强Agent依赖**预定义指令和工具**进行记忆更新。然而，语言模型可能缺乏确定**存储什么信息、如何结构化、何时更新**的能力——特别是当记忆系统变得复杂时。这导致次优的记忆构建和信息丢失。

### 9.2 基本假设

1. **RL可以教会Agent有效管理复杂记忆系统**：通过交互和反馈，Agent可以发现最优记忆策略
2. **下游QA准确率**是评估记忆质量的最佳信号：直接优化记忆构建的最终任务性能
3. **记忆架构应模块化**：RL框架与具体记忆设计解耦

### 9.3 遇到的困难

1. **监督信号缺失**：即使GPT-4o也无法正确使用复杂记忆工具，无法从现有模型获取可靠监督
2. **记忆操作组合爆炸**：多类型记忆×多种操作（insert/update/delete），决策空间巨大
3. **奖励设计复杂**：需要平衡准确性、压缩率、格式正确性和语义质量四个维度

### 9.4 算法解决方案

#### 9.4.1 任务设定

Agent处理对话序列 $\mathcal{C} = \{c_1, \ldots, c_n\}$。在步骤 $t$ 观察 $c_t$ 和当前记忆 $\mathcal{M}_{t-1}$，执行写操作序列：

$$a_t = (a_t^{(1)}, \ldots, a_t^{(K_t)})$$

其中 $a_t^{(k)} \in \mathcal{A}_{\text{write}} = \{\text{memory\_insert}, \text{memory\_update}, \text{memory\_delete}\}$

记忆更新：

$$\mathcal{M}_{t-1}^{(0)} = \mathcal{M}_{t-1}, \quad \mathcal{M}_{t-1}^{(k)} = T(\mathcal{M}_{t-1}^{(k-1)}, a_{t-1}^{(k)}), \quad \mathcal{M}_t = \mathcal{M}_{t-1}^{(K_t)}$$

#### 9.4.2 四维奖励函数

**1. 正确性奖励 $r_1$（全局）**：

$$r_1 = \frac{1}{m} \sum_{j=1}^{m} \mathbb{I}[\text{metric}(\hat{r}_j, r_j)]$$

其中 $\hat{r}_j = g(q_j, \phi(\mathcal{M}_n, q_j))$ 是RAG管道的预测答案。

**2. 工具调用格式奖励 $r_{2,t}$（步骤级）**：

$$r_{2,t} = \frac{1}{K_t} \sum_{k=1}^{K_t} s(a_t^{(k)}), \quad s(a_t^{(k)}) \in \{0, 1\}$$

**3. 压缩奖励 $r_3$（全局）**：

$$r_3 = 1 - \frac{l_m}{l_c}$$

其中 $l_m$ 是记忆总长度，$l_c$ 是chunk总长度。

**4. 记忆内容奖励 $r_{4,t}$（步骤级）**：

$$r_{4,t} = \frac{1}{K_t} \sum_{k=1}^{K_t} v(a_t^{(k)}), \quad v(a_t^{(k)}) \in \{0, 1\}$$

使用Qwen3-32B作为Judge验证语义有效性。

**最终奖励**：

$$r_t = r_1 + r_{2,t} + \beta r_3 + \gamma r_{4,t}$$

默认 $\beta = 0.05$，$\gamma = 0.1$。

#### 9.4.3 GRPO策略优化

优势计算：

$$A_t = \frac{r_t - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}$$

优化目标：

$$\mathcal{J}(\theta) = \mathbb{E}_{\mathcal{C} \sim P(\mathcal{C}), \mathcal{A} \sim \pi_{\text{old}}(\cdot|\mathcal{C},\mathcal{M}_0)} \sum_{t=1}^{n} \left[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|a_t|} \sum_{j=1}^{|a_t|} \min\left(\frac{\pi_\theta(a_{t,j}|\cdot)}{\pi_{\text{old}}(a_{t,j}|\cdot)} A_t, \text{clip}(\cdot) A_t\right)\right]$$

去除KL项以鼓励策略探索。

#### 9.4.4 记忆架构实例

三组件记忆：
1. **Core Memory**：持久文本摘要（最大512 token），始终在Agent上下文中
2. **Semantic Memory**：离散事实陈述集合，可独立检索和更新
3. **Episodic Memory**：带时间戳的事件集合

### 9.5 实验安排与结果

**训练**：Qwen3-4B为backbone，32×H100 GPU训练3天，562个实例，205步

**验证集结果**：

| 方法 | 平均性能 | 平均记忆(tokens) |
|------|---------|-----------------|
| Long-Context | 0.588 | 10.8K |
| RAG-Top2 | 0.567 | 11.3K |
| MemAgent | 0.236 | 0.84K |
| MEM1 | 0.111 | 0.17K |
| **Mem-α** | **0.642** | **7.9K** |

**MemoryAgentBench（OOD测试）**：

| 方法 | AR | TTL | LRU | 平均 |
|------|-----|-----|-----|------|
| Long-Context | 0.280/0.270/0.292 | 0.640/0.740/... | 0.125 | 0.461 |
| **Mem-α-4B** | **0.740/0.680/0.520** | **0.710/0.710/...** | **0.129** | **0.592** |

**关键发现**：
1. RL训练后的4B模型超越GPT-4.1-mini（0.642 vs 0.517）
2. 仅训练30K token以内的实例，可泛化到400K+ token序列（13×训练长度）
3. 记忆内容奖励 $r_4$ 对训练至关重要：$\gamma=0$ 导致灾难性性能退化

### 9.6 欠缺与不足

1. **训练数据规模小**：仅562个实例，可能限制策略多样性
2. **基础模型较小**：使用Qwen3-4B，8B模型反而出现指令遵循问题
3. **评估缺乏Agent场景**：主要在记忆构建/QA任务上评估，未涉及工具使用或深度研究
4. **核心记忆容量限制**：Core Memory限制512 token，可能不适合需要大量核心信息的场景
5. **未处理冲突解决**：训练数据排除了冲突解决维度
6. **RAG管道固定**：检索和生成组件在训练中固定，仅优化写策略，限制了端到端优化

---

## 综合对比与分析

### 方法论分类

| 类别 | 方法 | 核心思想 | 训练方式 |
|------|------|---------|---------|
| **上下文摘要** | ReSum | 外部摘要工具周期性压缩 | GRPO + 优势广播 |
| **上下文摘要** | SUPO | 端到端摘要策略优化 | 策略梯度（端到端） |
| **上下文折叠** | Context-Folding | 分支-折叠子轨迹 | FoldGRPO + 过程奖励 |
| **上下文折叠** | AgentFold | 多尺度主动上下文雕刻 | SFT（拒绝采样） |
| **迭代重构** | IterResearch | MDP工作空间重构 | EAPO（折扣奖励） |
| **记忆系统** | MemOS | OS级记忆管理 | 无RL（系统设计） |
| **记忆系统** | MIRIX | 六类型多Agent记忆 | 无RL（指令驱动） |
| **记忆系统** | EverMemOS | 自组织记忆+超图 | 无RL（工程框架） |
| **记忆+RL** | Mem-α | RL学习记忆构建策略 | GRPO（四维奖励） |

### 关键指标对比（BrowseComp）

| 方法 | 模型规模 | BrowseComp | 上下文效率 |
|------|---------|-----------|----------|
| ReSum-GRPO | 30B-A3B | 18.3% | 2×基线 |
| IterResearch | 30B-A3B | 37.3% | O(1)常数 |
| AgentFold | 30B-A3B | 36.2% | ~7K@100步 |
| OpenAI DeepResearch | 商用 | 51.5% | 未知 |

### 核心启示

1. **范式比规模更重要**：30B模型通过正确的上下文管理范式可超越671B模型
2. **RL是上下文管理优化的关键**：ReSum、IterResearch通过RL显著提升性能，AgentFold仅用SFT也展现了范式潜力
3. **迭代重构 vs 摘要压缩**：IterResearch的O(1)工作空间优于ReSum的周期摘要，在极长探索（2048步）上表现突出
4. **记忆系统需要RL训练**：Mem-α证明即使GPT-4o也无法仅通过指令有效利用复杂记忆工具
5. **未来方向**：将Agent工具使用的上下文管理（ReSum/IterResearch/AgentFold）与结构化记忆系统（MemOS/MIRIX/Mem-α）结合，可能是突破方向
