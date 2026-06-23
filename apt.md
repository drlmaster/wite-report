## §1 一句话定位

APT 是一种 **连续动作 VLA 的动作专家预训练方法**，不是新基础模型。它解决的是：连续动作 expert 随机初始化时容易走视觉捷径、破坏 VLM 语言能力，导致 OOD 指令泛化差。与视觉—语言—动作对齐相关度：**高**。本质上是 **新训练流程 + 新 action expert 融合结构**。【明示：Abstract / Fig.1 / Fig.2】

## §2 模型框架与网络架构

### 2.1 总体数据流

```text
第三视角/腕部视觉输入 + 语言指令 + 当前 proprioception / 历史动作
→ Qwen3-VL 编码视觉与语言 token
→ MLP projector 投入 action expert token 空间
→ Transformer diffusion action expert 自注意力建模
→ 去噪生成未来 action chunk
→ 连续末端动作
```

结论：APT 属于 **VLM + diffusion action expert** 的双系统 VLA。视觉语言 backbone 明确为 **Qwen3-VL-2B-Instruct**；action expert 是 20 层 Transformer diffusion model；视觉 token 与语言 token 先经独立 MLP projector，再进入 action expert embedding space；action encoder / decoder 是 2-layer MLP，hidden dim 768。【明示：Appendix A】

动作 head 不是离散 token，也不是接入 LLM 词表，而是连续动作 diffusion expert。动作表示为 SE(3) 上的 10 维末端动作：3D translation + 6D rotation + normalized gripper width。action chunk length 为 **32**，history action length 为 **1**；训练采用 DDPM 100 diffusion steps，推理采用 DDIM 20 denoising steps。【明示：Appendix A】

模块训练方式是两阶段。Stage 1 冻结 VLM，只用视觉—动作对训练 action expert 的 VA prior；Stage 2 注入语言 token，并联合训练完整 VLA policy；之后再做任务特定 finetuning。【明示：Fig.2 / §3.2】

### 2.2 闭环方式

是否重新读取最新观测：**【未知】论文未明确报告部署时每次 action chunk 后是否重新观测**。
是否 receding horizon：**【未知】**。
chunk 每次执行多少步：action chunk length = 32，但每次实际执行多少步未报告。
失败检测或恢复机制：**【未知】未报告显式失败检测/恢复机制**。

最终判定：**action-chunk policy，但闭环执行细节未知**。不能把行为克隆训练或 action chunk 生成直接等同于完整轨迹开环。

## §3 核心创新点

**创新 1：把连续动作 VLA 的语言泛化问题归因到 action expert 随机初始化，而不只是 VLM 被破坏。**
结论：这个切入点成立，有一定审稿价值。论文认为 VLA 数据中一个语言指令对应大量视觉—动作帧，语言多样性远小于视觉动作多样性，导致随机初始化的 action expert 倾向依赖视觉捷径，并产生噪声梯度腐蚀 VLM。【明示：Abstract / §1 / §3.1】

**创新 2：Bayesian factorization 引出 VA prior → VLA likelihood 的两阶段训练。**
结论：这是论文最核心的机制。作者将策略分解为 π(a|v,l) ∝ πp(a|v)·L(l|v,a)，先在视觉—动作对上训练语言无关 VA prior，再引入语言做 VLA likelihood alignment。这个设计把“先学会怎么动，再学语言如何指定动作”形式化了。【明示：Eq.2 / §3.2 / Fig.2】

**创新 3：Layer-wise gated fusion 让 VLM 特征逐层注入 action expert。**
结论：这是架构层面的辅助创新。APT 从 Qwen3-VL 均匀采样中间层特征，并通过可学习 sigmoid gate 注入 action expert 每层，使浅层空间信息与深层语义信息都能进入动作专家，同时尽量保留 VA prior。【明示：Fig.3 / Eq.3 / §3.3】

## §4 实验结论与证据强度

结论：性能提升主要来自 **训练流程中的 action expert pretraining**，架构 gated fusion 和大规模预训练数据进一步放大收益。

LIBERO-PRO 中，OpenVLA 和 π0 在所有 Pos / Task 扰动上为 0；π0.5 平均 11；LangForce 平均 14；APT 平均 19；APT + Ft VLM 平均 27。这个结果支持作者的主张：仅靠 KI stop-gradient 不足以解决 OOD 指令泛化，动作专家初始化更关键。【明示：Table 1】

Rigid Object Pick-Place 中，带 2-stage 的 APT 明显优于不带 2-stage 的变体。例如 APT + 2-Stage + Ft VLM 在 SO/UO/UC/UOUE 上为 98/84/92/58，而 π0.5 为 84/70/86/50。这里最能证明“收益来自 action expert pretraining”，因为表中直接控制了 KI、2-Stage、Ft VLM 三个因素。【明示：Table 2】

真实实验中，APT 在 Pick-Place 和 Clutter Pick-Place 上均优于 π0.5。Pick-Place 总体从 63/110 提升到 90/110；Clutter Pick-Place 从 43/80 提升到 60/80。组合任务 chaining 中，π0.5 基本崩溃，而 APT 仍保持较高成功率，说明语言结构解析能力确实有改善。【明示：Table 3 / Fig.6 / Fig.7】

## §5 与最接近方法比较

最接近的是 **π0.5 / KI 系列** 和 **BayesVLA**。APT 与 π0.5 的差异不是动作生成范式，而是训练哲学：π0.5 主要通过 Knowledge Insulation 保护 VLM，APT 则认为根因在 action expert 未预训练，因此先初始化 action expert 的 VA prior。与 BayesVLA 相比，APT 继承 Bayesian factorization 思想，但没有依赖 pre-/post-contact 分解，而是转成更通用的 action expert pretraining，因此更适合大规模异构数据。【明示：§2 / §3.2 / §4.1.3】

## §6 核心缺点

**缺点 1：闭环部署细节不足。** action chunk 长度、推理 diffusion steps 报告了，但 receding horizon、每次执行步数、观测刷新频率、失败恢复机制未明确。【未知】

**缺点 2：长程记忆仍弱。** 作者自己承认当前设计没有显式 long-horizon memory，限制需要多步进度追踪的任务泛化。【明示：Limitations】

**缺点 3：实验场景仍集中在桌面操作。** 真实实验主要是 pick-place、clutter pick-place、table storage + pick-place；移动操作、全身控制、双臂复杂协作没有充分验证。【明示：§4.2 / Limitations】

## §7 是否值得精读、复现或借鉴

结论：**值得精读，尤其适合你当前 GR00T / VLA 微调方向借鉴。**

最值得借鉴的是它的“先 VA、后 VLA”训练逻辑。对你的 BTA / 双臂服务任务，可以转化为：先用视觉 + 状态 + 动作训练双臂动作先验，让模型先学会抓取、递送、倒水、放杯等动作流形；再注入语言、phase prompt、role token 做任务对齐。它比直接把所有语言和动作混在一起训更有解释力，也更容易包装成“动作先验稳定化 + 语言任务对齐”。

最终判断：APT 的创新不是提出一个全新 VLA 基座，而是指出连续动作 VLA 的一个关键训练病灶：**动作专家没有预训练会拖累语言泛化**。这个判断有理论分解、消融和真实实验支撑，核心贡献成立。
