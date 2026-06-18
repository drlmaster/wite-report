# MemoryVLA 论文阅读笔记

论文：**MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation**  
来源：`memoryVLA.pdf`，ICLR 2026 conference paper  
核心主题：在 VLA 模型中显式加入感知-认知记忆，用于长程、非马尔可夫的机器人操作任务。

## 1. 一句话总结

MemoryVLA 的核心贡献是：不再让 VLA 只依赖当前单帧观测，而是构建一个类似人类“工作记忆 + 情景记忆”的结构，将当前视觉/语言表征与历史关键上下文进行检索、融合和压缩更新，从而提升长程操作、重复计数、隐藏状态推理和真实机器人任务中的决策稳定性。

## 2. 问题动机

主流 VLA 模型如 OpenVLA、pi0、CogACT 等通常只根据当前图像和语言指令输出动作。这在短程操作中可行，但很多机器人任务本质上是非马尔可夫的：当前画面不足以判断下一步动作，必须知道过去发生过什么。

论文给出的典型例子是按按钮任务：按钮被按前和按后在视觉上几乎没有明显差异，如果模型不知道“刚才是否已经按过”，就容易重复执行或跳过关键步骤。

直接拼接多帧历史图像也不是理想方案：

- VLM 自注意力复杂度随 token 数增长，长历史会带来较高计算成本。
- 多帧输入与很多 VLA 的单帧预训练分布不一致，可能破坏已有视觉-语言能力。
- 简单历史堆叠没有区分“低层视觉细节”和“高层任务语义”，容易把冗余帧也塞进上下文。

因此，论文的关键判断是：VLA 需要一种更结构化的时间记忆机制，而不是简单增加输入帧数。

## 3. 方法框架

MemoryVLA 被组织为 **Cognition-Memory-Action** 框架，包含三个主要部分。

### 3.1 Vision-Language Cognition Module

模型基于 7B Prismatic VLM。视觉侧使用 DINOv2 和 SigLIP 并行编码 RGB 图像，再通过压缩模块得到感知 token；语言侧将视觉特征和指令送入 LLaMA-7B，并取 EOS 位置输出作为认知 token。

这里有一个重要设计：论文把表征分成两类。

- **Perceptual tokens**：保存低层视觉细节，如物体位置、外观、空间关系。
- **Cognitive token**：保存高层语义摘要，如任务意图、阶段状态、语言相关决策信息。

这两类 token 共同组成当前时刻的工作记忆。

### 3.2 Perceptual-Cognitive Memory Bank

论文提出 **Perceptual-Cognitive Memory Bank, PCMB**，分别存储历史感知信息和历史认知信息。它不是无脑保存所有帧，而是维护一个有限长度的记忆库。

每个时间步都会执行三件事：

- **Memory Retrieval**：当前工作记忆作为 query，从 PCMB 中检索与当前决策相关的历史信息，并加入时间位置编码。
- **Memory Gate Fusion**：使用门控机制融合当前 token 和检索到的历史 token，让模型自己决定该相信当前观察还是历史记忆。
- **Memory Consolidation**：当记忆库达到容量上限时，合并时间上相邻且语义相似的记忆，减少冗余并保留关键信息。

这个设计对应了人类认知中的两个层次：工作记忆用于即时控制，长期记忆用于保存过去经验，并在需要时被检索。

### 3.3 Memory-Conditioned Action Expert

动作头采用 diffusion transformer。模型输出长度为 16 的未来动作 chunk，每个动作包含 7-DoF 控制量。

动作生成时，认知 token 通过 cognition attention 提供高层语义约束，感知 token 通过 perception attention 补充细粒度视觉信息。这样动作专家不仅知道“要做什么”，也知道“具体在哪里做”。

## 4. 主要创新点

### 创新点 1：将 VLA 的时间建模从“多帧输入”转为“记忆系统”

已有方法常见做法是拼接历史帧、滑动窗口、多帧特征聚合或把历史轨迹画在图像上。MemoryVLA 的创新在于把历史上下文显式建模为一个可检索、可更新、可压缩的记忆库。

这种设计的价值是：长程历史不再只是额外输入，而是一个有组织的状态系统。

### 创新点 2：同时保存低层感知细节和高层认知语义

论文没有只保存视觉帧特征，也没有只保存语言/语义摘要，而是将历史信息分成 perceptual memory 和 cognitive memory。

消融结果表明，两者结合最好：在 SimplerEnv-Bridge 上，单独 cognitive memory 为 63.5%，单独 perceptual memory 为 64.6%，二者结合达到 71.9%。

这个结果说明，机器人长程操作既需要记住“发生了什么”，也需要记住“物体当时在哪里、是什么状态”。

### 创新点 3：用门控融合解决“当前观测 vs 历史记忆”的权重问题

模型不是简单把历史信息加到当前表征上，而是学习一个 gate：

- 当前图像可靠时，可以更多依赖当前观测。
- 当前图像存在遮挡、状态不可见或语义歧义时，可以更多依赖历史记忆。

消融中，gate fusion 在 SimplerEnv-Bridge 上达到 71.9%，简单 add 只有 67.7%；在真实 Clean Table & Count 任务上，gate 为 84%，add 为 78%。

### 创新点 4：记忆压缩采用相邻相似记忆合并，而不是 FIFO 丢弃

MemoryVLA 在记忆满时不直接删除最早条目，而是计算相邻记忆的相似度，合并最相似的一对。

这个思路很实用：机器人轨迹中大量连续帧是冗余的，合并相似邻居可以保留阶段性信息，同时控制计算成本。消融中 token merge 明显优于 FIFO。

### 创新点 5：把认知科学叙事转化为可实现的模型结构

论文从工作记忆、海马体情景记忆、verbatim/gist representation 等认知科学概念出发，但最终落到了具体模块：工作记忆、PCMB、检索、融合、合并、动作专家。

这类写法很值得借鉴：不是泛泛地说“受到人类启发”，而是把类比关系映射到架构组件和实验验证上。

## 5. 实验结果与证据

论文评估覆盖 3 类机器人、6 个 benchmark、150+ 任务和 500+ variations。

关键结果如下：

- SimplerEnv-Bridge：MemoryVLA 平均成功率 71.9%，比 CogACT-Large 高 14.6 个点。
- SimplerEnv-Fractal：平均 72.7%，比 CogACT 高 4.6 个点。
- LIBERO：五个 suite 平均 96.5%，比 CogACT 高 3.3 个点。
- Mikasa-Robo：平均 41.2%，比 pi0 高 11.8 个点，在 ShellGameTouch 上提升尤其明显。
- 真实机器人 general tasks：平均 85%，比 CogACT 高 9 个点。
- 真实机器人 long-horizon temporal tasks：平均 83%，比 CogACT 高 26 个点。

最有说服力的是长程真实任务结果。MemoryVLA 在 Seq. Push Buttons、Change Food、Guess Where、Clean Table & Count 等任务上大幅领先，说明记忆机制确实解决了当前视觉不足以决策的问题。

## 6. 有价值的可借鉴思路

### 6.1 在我们的系统中引入“任务阶段记忆”

如果我们的机器人系统包含饮品服务、双臂协作、桌面清理或多步骤装配，那么当前画面往往无法表达完整状态。例如：

- 哪个杯子已经放过冰块？
- 哪个按钮或容器已经操作过？
- 用户点单中的哪一步已经完成？
- 双臂中左臂/右臂各自完成了哪个子任务？
- 某个物体短暂被遮挡后，是否还能保持身份一致？

可以设计一个轻量级任务阶段记忆，将关键事件写入 memory bank，而不是让策略完全从当前图像推断。

### 6.2 将记忆分为“感知记忆”和“语义记忆”

这对具身系统尤其有用。感知记忆可以保存物体位置、姿态、状态、抓取点、容器位置等；语义记忆保存任务阶段、用户意图、已完成动作、失败重试记录等。

对于论文写作，这也可以成为一个清晰的模块叙事：

- 感知记忆解决视觉遮挡和空间状态遗忘。
- 语义记忆解决任务阶段、顺序约束和目标一致性。
- 二者融合支撑长程操作决策。

### 6.3 不必一开始就做大模型级记忆，可以先做事件级 memory

MemoryVLA 的机制比较重，但思想可以简化。工程上可以先保存结构化事件：

- `object_seen(object, pose, time)`
- `action_done(skill, target, result, time)`
- `subgoal_completed(subgoal, time)`
- `user_preference(item, constraint, time)`
- `failure_retry(skill, reason, time)`

然后在决策前检索最近相关事件，作为额外状态输入给 planner 或 policy。

### 6.4 对长程任务，记忆长度应该随任务时长调整

论文附录说明，真实 long-horizon temporal tasks 的动作长度明显更长，因此需要更大的 memory length。Clean Table & Count 任务中 memory length 256 最优，64 和 512 都更差。

这提醒我们：记忆不是越长越好。过短会遗忘关键阶段，过长会引入干扰和计算负担。可以根据任务粒度设置不同 memory capacity。

### 6.5 用“相邻相似合并”替代“先进先出删除”

对于连续机器人轨迹，很多历史状态高度相似。与其丢掉最早状态，不如合并相邻且相似的状态。这种方式适合写成一个轻量 memory consolidation 模块，也容易做消融。

可借鉴写法：

> 当记忆容量达到上限时，系统优先合并时间相邻且语义相似的记忆条目，从而保留阶段转折点并压缩冗余连续状态。

### 6.6 论文叙事可以围绕“非马尔可夫性”展开

MemoryVLA 最强的故事不是“我们用了 memory”，而是“机器人长程操作不是马尔可夫过程，当前观测不足以决策，因此需要历史状态”。

如果我们要写自己的 VLA/双臂服务系统论文，也可以采用类似论证：

1. 真实服务任务存在阶段依赖、遮挡、重复计数和多对象身份保持。
2. 单帧 VLA 或单步策略无法可靠区分这些隐含状态。
3. 因此需要显式维护任务记忆，并让动作决策受记忆条件约束。

## 7. 可以迁移到自己工作的模块设计

一个可落地的简化版本如下：

```text
Observation + Instruction
        |
        v
Perception Encoder ----> Perceptual Memory
        |
        v
Task Parser / VLM -----> Semantic Memory
        |
        v
Memory Retrieval
        |
        v
State-Aware Planner / Action Policy
        |
        v
Action + Memory Update
```

可定义三类 memory item：

- **Visual state memory**：物体位置、容器状态、遮挡前观测、双臂末端位置。
- **Task progress memory**：已完成步骤、当前子目标、剩余子目标、计数状态。
- **Interaction memory**：用户指令、偏好、异常反馈、失败重试记录。

这样既保留 MemoryVLA 的核心思想，又不需要完全复现其 7B VLM + diffusion action expert。

## 8. 局限与需要谨慎的地方

MemoryVLA 虽然结果强，但仍有一些风险点：

- 模型规模较大，完整复现需要较高训练资源。
- 记忆机制依赖高质量表征，若视觉编码器或 VLM 对场景理解错误，记忆也会写入错误信息。
- 在严重视角变化下泛化仍会明显下降，例如仿真中的 unseen camera view。
- 记忆检索的可解释性主要通过案例展示，系统性分析还不够充分。
- 真实任务虽然覆盖 12 个任务，但与开放世界服务场景相比仍偏受控。

## 9. 对我们写论文的启发

如果要把这类思想写进自己的论文，可以重点借鉴以下表达方式：

- 用“非马尔可夫长程操作”作为问题定义，而不是泛泛说“长任务很难”。
- 将记忆拆成不同层级，形成清晰模块边界。
- 每个模块都对应一个具体失败模式：遗忘、遮挡、重复动作、阶段混淆、长历史冗余。
- 消融不要只验证有没有 memory，还要验证 memory 类型、长度、融合方式和更新方式。
- 真实机器人实验最好设计专门依赖历史的任务，而不是只跑普通 pick-and-place。

## 10. 最值得吸收的核心观点

MemoryVLA 最有价值的观点是：**对于长程机器人操作，历史不是额外上下文，而是决策状态的一部分。**

这意味着 VLA 系统不应只追求更强的单帧视觉语言理解，还需要显式维护“过去发生过什么、当前处于哪个阶段、哪些信息虽然不可见但仍然有效”。这对服务机器人、双臂协作、桌面整理、按顺序执行和人机交互任务都很重要。
