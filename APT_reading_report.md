# APT 论文阅读报告

论文：**APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies**  
作者：Kechun Xu, Zhenjie Zhu, Anzhe Chen, Rong Xiong, Yue Wang  
来源：`APT.pdf`，arXiv:2606.12366v1，2026-06-10  
主题：连续动作 VLA 策略的动作专家预训练与指令泛化。

## 1. 一句话总结

APT 的核心观点是：连续动作 VLA 模型的动作专家如果从随机初始化开始在语言稀疏、视觉动作密集的数据上联合训练，容易学到绕开语言的视觉捷径，并向 VLM 反传有害梯度；因此应先把动作专家预训练成一个不依赖语言的 Vision-Action prior，再引入语言进行 VLA 对齐，从而提升未见指令、未见物体和组合任务的泛化能力。

## 2. 论文要解决的问题

现有 VLA 模型通常把预训练 VLM 与连续动作专家结合起来。VLM 负责视觉语言表征，动作专家负责生成连续机器人动作。这类方法在动作质量上比离散 action token 更适合灵巧操作，但在语言泛化上存在明显短板：模型在训练分布内表现较好，一旦遇到改写指令、未见物体、目标替换或多任务组合，就容易失败。

论文认为根源不是单纯的模型容量不足，而是 VLA 数据的结构性不平衡。一个机器人轨迹往往包含大量视觉帧和动作标注，却只共享一个语言指令。视觉和动作变化远比语言丰富，随机初始化的动作专家在训练早期更容易依赖视觉线索预测动作，而不是真正使用语言。这样会形成两类问题：动作专家学到 visual shortcut；动作专家向 VLM 反传噪声梯度，破坏 VLM 原本的语言能力。

这也解释了为什么 Knowledge Insulation 这类 stop-gradient 方法能缓解问题但不充分。它能保护 VLM 不被动作损失污染，却没有让动作专家本身获得更好的语言对齐初始化。APT 的切入点是动作专家初始化：先让动作专家学会稳定的视觉运动先验，再引入语言条件。

## 3. 方法框架

APT 从贝叶斯分解出发，将 VLA 策略写成：

```text
pi(a | v, l) proportional to pi_p(a | v) * L(l | v, a)
```

其中 `pi_p(a | v)` 是 Vision-Action prior，只根据视觉观测预测动作；`L(l | v, a)` 是语言条件下的 VLA likelihood，用语言把动作分布约束到具体任务上。

这个分解带来一个两阶段训练流程。

### 3.1 Stage 1：VA Prior Pretraining

第一阶段只训练动作专家的视觉动作能力。模型使用 VLM 提供的视觉 token，但冻结 VLM 参数，并完全 mask 掉语言 token。动作专家只看到视觉和动作，学习 `pi_p(a | v)`。

论文的关键判断是：视觉-动作对本身是平衡的，每个视觉帧都有对应动作，不会像 VLA triplet 那样因为一个语言指令对应大量帧而诱导语言忽略。因此，Stage 1 中的 vision-only 学习不是有害捷径，而是有意构建的视觉运动先验。

从训练形式上看，APT 的 Stage 1 本质上就是视觉条件下的模仿学习，和 ACT、Diffusion Policy 等 visuomotor imitation learning 方法很接近：输入图像、本体状态或历史动作，监督信号是专家动作或 action chunk，训练目标是让策略学会 `pi(a | v)`。因此，APT 并不是发明了“用视觉预测动作”这件事。

APT 的关键在于它把这种模仿学习放进了 `VLM + continuous action expert` 的 VLA 训练流程中，作为 action expert 的初始化阶段。ACT 中的视觉模仿学习通常就是最终策略；而 APT Stage 1 中的视觉模仿学习只是为后续 VLA 训练提供一个 VA prior。也就是说，APT 先让动作专家学会“看图怎么动”，再在 Stage 2 中引入语言，让语言去调制已经形成的动作先验。

这也解释了为什么 Stage 1 的 `pi(a | v)` 不应被称为视觉捷径。视觉捷径指的是：模型明明在 VLA 训练中接收了语言输入 `pi(a | v, l)`，但实际决策时绕开语言、退化成 `pi(a | v)`。Stage 1 则是有意不输入语言，目标本来就是训练视觉动作先验。简单说，只给模型看图做动作是 VA pretraining；给了语言却不用语言，才是 visual shortcut。

### 3.2 Stage 2：VLA Likelihood Alignment

第二阶段再引入语言 token，让模型学习完整的 VLA 策略。APT 不是简单把语言 token 插入已经训练好的动作专家，而是在网络中插入新的 attention 层，并移除语言 mask。继承自 Stage 1 的层保留 VA prior，新插入的层主要承担语言对齐。

论文强调，Stage 2 并不冻结 VA prior，而是在完整预训练数据上联合优化所有层。这一点不同于 BayesVLA 中冻结 prior 的思路。APT 的假设是：在大规模数据上，让先验和语言 likelihood 共同调整，可以得到更好的平衡。

### 3.3 Action Expert 设计

APT 的动作专家是 Transformer-based diffusion model。动作 token 由三部分组成：历史动作、当前本体状态、待去噪的 noisy action。去噪 timestep 通过 FiLM 注入每个 attention 层。

VLM backbone 使用 Qwen3-VL-2B-Instruct。为了把 VLM 表征注入动作专家，APT 设计了 layer-wise gated fusion：从 Qwen3-VL 的不同深度均匀采样中间特征，并通过可学习 gate 注入动作专家每一层。形式上可以理解为：

```text
h_in(i+1) = h_out(i) + sigmoid(w_i) * phi_i(v, l)
```

这个设计有两个目的：浅层 VLM 特征提供空间和视觉细节，深层 VLM 特征提供语义信息；gate 控制 VLM 特征对动作专家的影响，避免语言注入过猛而破坏已学到的 VA prior。

附录还给出若干实现细节：动作表示为 10 维 SE(3) 动作，包括 3D 平移、6D 连续旋转和归一化夹爪宽度；动作统一投影到相机坐标系以增强 embodiment equivariance；动作专家有 20 个 attention 层，action chunk 长度为 32；Stage 1 和 Stage 2 各训练 100k iterations，batch size 为 256。

## 4. 主要创新点

第一，论文把连续动作 VLA 的语言泛化问题明确归因到动作专家初始化和数据不平衡的共同作用。已有工作多强调 VLM 被动作训练污染，APT 进一步指出：污染来自随机初始化动作专家在不平衡 VLA 数据上先学到视觉捷径。

第二，APT 将 BayesVLA 式的 VA prior / VLA likelihood 分解转化为可扩展的动作专家预训练流程。它不需要额外构造新的语言数据，而是直接从已有 VLA 数据中取视觉-动作对进行 Stage 1 训练。

第三，论文提出了更适合保留先验的语言注入方式。直接 token insertion 会突然改变 Stage 1 中学到的动作分布，APT 通过新插入层和 gated fusion 让语言逐步调制动作先验。

第四，APT 不是绑定某一个架构。论文把两阶段动作专家预训练应用到 pi-style、GR00T-style 和作者自己的 APT action expert 上，显示该思路对不同连续动作 VLA 架构都有一定通用性。

## 5. 实验结果与证据

### 5.1 LIBERO-PRO

LIBERO-PRO 用位置扰动和任务指令扰动评估语言泛化。结果显示，OpenVLA 和 pi0 在该 benchmark 上几乎完全失败，表明直接联合训练容易依赖训练分布中的视觉捷径。pi0.5 通过 Knowledge Insulation 在位置扰动上有所恢复，但在任务指令替换上仍接近失败。

关键结果如下：

- OpenVLA：平均成功率 0%。
- pi0：平均成功率 0%。
- pi0.5：平均成功率 11%。
- LangForce：平均成功率 14%。
- APT：平均成功率 19%。
- APT with VLM finetuning：平均成功率 27%。

这个结果支持论文的核心判断：只保护 VLM 不够，动作专家本身需要更好的 VA 初始化。APT with VLM finetuning 进一步说明，在动作专家初始化足够好时，联合微调 VLM 不一定有害，反而可以继续提升语言泛化。

### 5.2 Rigid Object Pick-Place

Rigid Object Pick-Place 包含 SO、UO、UC、UOUE 四个设置，分别考察已见物体、未见物体、未见容器、未见物体加未见环境。

表 2 中，pi0 的四项成功率为 42、30、26、16；pi0.5 提升到 84、70、86、50。APT 的不同变体显示了两阶段预训练的贡献：带 Knowledge Insulation 和 2-stage 的 APT 达到 96、74、90、62；保留 2-stage 并联合 finetune VLM 的版本达到 98、84、92、58。

这里最重要的结论不是单个数字领先，而是消融逻辑：没有 2-stage 时，仅 stop-gradient 并不能充分解决泛化失败；有 2-stage 后，即使不用额外 VL reasoning 数据，也能超过 pi0.5。

### 5.3 架构消融

论文把动作专家预训练应用到三类结构：pi-style、GR00T-style 和 APT action expert。Figure 4 显示，2-stage training 在几乎所有设置中提升泛化，尤其在 GR00T-style 和 APT action expert 上提升更明显。

论文给出的解释是：GR00T-style 只用最终 VLM 层特征通过 cross-attention 条件化动作专家，APT 通过 gated fusion 注入 VLM 特征，这两种方式都更容易保留 Stage 1 学到的 VA prior。相比之下，pi-style 在每层直接混合 VLM 和 action expert token，更容易破坏预训练动作表征。

### 5.4 大规模预训练与语言注入消融

论文比较了不使用大规模预训练、直接 token insertion 和完整 APT。结果显示，只在任务数据上做两阶段训练也能带来一定泛化，说明动作专家预训练本身是一种有效 inductive bias。但在 UO 和 UOUE 这类涉及未见类别和未见环境的设置中，大规模异构数据预训练显著更强。

语言注入方面，gated fusion 明显优于 token insertion。原因是 token insertion 会在 Stage 2 突然改变已学到的 VA 分布，导致部分遗忘；gated fusion 则以更可控的方式调制先验。

### 5.5 真实机器人实验

真实机器人使用 Agilex Cobot 平台、Piper 机械臂和两个 ORBBEC DaBai RGB-D 相机。每个任务只收集 30 条 tele-operated demonstrations 进行 finetune，并与 pi0.5 比较。

单任务 Pick-Place 中，APT 在总成功数上为 90/110，pi0.5 为 63/110。按主文表 3，APT 在 SO、UO、UOUC、UOUCUE 上分别为 29/30、17/20、16/20、28/40；pi0.5 分别为 27/30、11/20、9/20、16/40。

Clutter Pick-Place 中，APT 为 60/80，pi0.5 为 43/80。主文表 3 中，APT 在 SO、UC、UO、UOUE 上分别为 25/30、22/30、7/10、6/10；pi0.5 分别为 18/30、18/30、4/10、3/10。论文案例显示，pi0.5 更容易把颜色相似的物体混淆，也更容易在 push-to-grasp 转换处停留。

组合任务部分最有启发：当两个任务以 separate prompts 的 task coaching 方式给出时，两种方法都表现较好；当两个任务被拼接成一个 chained prompt 时，pi0.5 几乎崩溃，APT 仍能保持较强表现。这说明差距不只是单条指令理解，而是多子任务语言解析和任务切换。

## 6. 与相关工作的关系

APT 与 Knowledge Insulation 是互补关系。Knowledge Insulation 主要保护 VLM，避免动作损失破坏语言表征；APT 主要改善动作专家初始化，使动作专家先掌握稳定 VA prior，再学习语言对齐。论文结果显示，只有 KI 不足以解决 OOD task generalization，而有 VA prior 后，联合训练 VLM 也可以变成有利因素。

APT 与 BayesVLA 的关系更直接。BayesVLA 提供 VA prior / VLA likelihood 的贝叶斯诊断，但其 pre-/post-contact 分解限制了异构数据扩展。APT 把这一思想泛化为动作专家预训练，不再依赖特定接触阶段分解，因此更适合大规模 VLA 数据。

APT 与 LangForce 的目标相近，都试图增强语言对动作的约束。LangForce 通过最大化动作和语言的条件互信息强化语言依赖，但可能在布局变化上牺牲视觉适应。APT 则试图从初始化上同时保留视觉运动能力和语言对齐能力。

## 7. 对当前 BTA/VLA 项目的启发

这篇论文对我们的双臂饮品服务 VLA 项目有直接启发，但需要注意：APT 解决的是通用连续动作 VLA 的动作专家预训练问题，而 BTA 的核心应继续定位为轻量的双臂任务结构适配层，不应把贡献改写成新的通用 VLA backbone。

### 7.1 可借鉴“先结构化动作，再语言对齐”的训练叙事

BTA 当前强调 task phase、arm role、左右臂 action subspace 和 role-aware loss。APT 的启发是：在引入复杂语言和任务组合之前，可以先让动作模块获得更稳定的动作先验。对双臂饮品服务而言，这个先验不一定是纯视觉动作 prior，也可以是 phase-aware / role-aware 的动作先验。

一个适合本项目的表述是：先通过阶段和角色约束降低双臂动作学习的不确定性，再让语言或任务提示调制具体动作。这样可以避免把所有复杂性都交给 VLM 和动作头端到端学习。

### 7.2 可设计“角色动作 prior”的消融

APT 证明了 VA prior 的价值。对 BTA 来说，可以对应设计 role-conditioned action prior 或 phase-conditioned action prior：

- 不加 phase/role，直接全轨迹 imitation。
- 加 phase token，但不做左右臂分区损失。
- 加 role token 和 action subspace loss。
- 加 role-aware weighted loss。

如果实验资源允许，还可以进一步比较“先训练单臂/子技能动作先验，再全流程微调”和“直接全流程微调”。这能把 APT 的两阶段思想转化为更贴合双臂服务任务的训练设计。

### 7.3 对 GR00T-style 架构有提示价值

论文的架构消融显示，GR00T-style 在动作专家预训练后提升明显，原因可能是最终层 VLM feature 通过 cross-attention 条件化动作专家，更容易保留动作先验。这与我们使用 GR00T-style policy fine-tuning 的方向契合。

在写作中可以谨慎引用这一点：连续动作专家的初始化和语言注入方式会影响指令泛化；因此，BTA 选择在 base VLA/GR00T-style policy 之上加入轻量 task-role conditioning，而不是重写整个 backbone，是一种降低干扰、增强可诊断性的设计。

### 7.4 组合任务失败模式值得借鉴

APT 的真实实验显示，pi0.5 在 chained prompt 下会过度执行第一个任务，无法切换到第二个任务。这个失败模式与饮品服务中的多阶段任务高度相关。例如抓杯、对齐、倒饮料、放置、撤回之间也存在明确阶段切换。

因此，我们的论文可以把 phase prompt 和 task role token 的价值写得更具体：它们不是简单增加标签，而是给策略一个可观察的任务阶段接口，降低长程指令中子任务边界不清导致的动作混淆。

### 7.5 不要过度声称

APT 自己也承认没有显式 long-horizon memory，组合任务中仍会出现子任务终止检测失败。我们的 BTA 如果没有系统性长程记忆模块，也不应声称已经解决长程记忆问题。更稳妥的写法是：BTA 通过 phase/role 结构减少阶段混淆风险，提高诊断性；长程记忆和自动阶段终止检测仍是后续方向。

## 8. 局限与需要谨慎的地方

论文的主要局限是没有显式建模 long-horizon memory。APT 可以改善 chained prompt 的执行，但失败案例仍包括抓取后继续 push、完成放置但漏掉关盒动作。这说明动作专家预训练不能替代任务进度记忆。

真实实验仍集中在桌面操作，包括 pick-place、clutter pick-place 和 table storage + pick-place 组合。论文明确提到，移动操作、全身控制或更开放的服务场景仍未验证。

对比实验也存在一些实现差异需要谨慎解读。附录提到真实实验中 pi0.5 使用 joint-space actions，而 APT 使用 camera-frame action representation。虽然这是各自预训练设置决定的，但它可能影响真实机器人对比的纯粹性。

复现成本较高。APT 使用 Qwen3-VL-2B-Instruct、20 层动作专家、两阶段各 100k iterations，并在 DROID、AgiBotWorld-Alpha、InternData-A1、InternVLA-M1 等大规模数据上预训练。对于小团队来说，完整复现比借鉴训练思想更现实。

论文引用了不少 2025-2026 年的并行或未来工作，若用于正式论文 related work，需要再次核对这些文献的公开状态和版本。

## 9. 阅读结论

APT 最值得吸收的不是某个具体网络细节，而是它对连续动作 VLA 语言泛化失败的诊断：问题不只在 VLM 是否保留语言能力，也在动作专家是否从一开始就被不平衡数据推向视觉捷径。通过先训练 VA prior，再做语言对齐，APT 将动作生成和语言 grounding 的学习难度拆开。

对我们的工作而言，这篇论文可以作为一个重要支撑：在 VLA/GR00T-style 策略中，动作专家或动作适配层的结构化初始化、任务阶段约束和角色条件化，都可能比单纯端到端微调更有利于组合任务和 OOD 指令泛化。BTA 的叙事可以沿着这个方向展开，但要保持项目自身贡献边界：APT 是通用 action expert pretraining，BTA 是面向双臂饮品服务的轻量 task-aware adaptation。
