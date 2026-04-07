## 强化学习

常用符号约定：
- $\pi$ = Policy，用于表示策略函数
- $\pi(s) = a$：表示确定性策略，在状态 $s\in S$ 下，策略 $\pi$ 会将其映射到动作 $a\in A$
- $\pi(a|s) = P(A=a|S=s)$：表示随机性策略，策略函数 $\pi$ 输出动作的概率

### 基于值函数的强化学习方法

所谓的值函数，实际上包括：
- **状态值函数 $V(s)$**：表示从状态 s 开始，以后得到的 reward 的期望值
- **动作价值函数 $Q(s, a)$**：表示在状态 s 下采取动作 a 获得的 reward

之所以称为基于值函数的强化学习，是因为我们**不直接学习策略 $\pi$（状态集 S 到动作集 A 的映射）**，而是**学习一个状态的值函数**（状态值或者动作值函数），策略通过这个值函数得到（一般是取使值函数最大的动作）

#### 贝尔曼方程

建立**当前**状态值函数和**未来**状态值函数的**递归关系**

- **状态值函数的贝尔曼方程**：$V(s) = \mathbb{E}_{a\sim \pi(·|s)}\mathbb{E}_{s', r\sim P}\big[r(s, a, s')+\gamma V(s')\big]$，$r$ 表示环境的**奖励模型**（$s',r\sim P$ 表示环境的随机性，环境只能给状态转移概率 $P(s'|s,a)$，所以有可能在同一个状态下采取相同的动作转移到的状态和得到的奖励不相同），用于评估动作的奖励，$s'$ 表示下一状态，$\gamma\in (0,1)$ 为折扣因子
- **动作价值函数的贝尔曼方程**：$Q(s,a)=\mathbb{E}_{s',r\sim P}\big[r(s,a,s')+\gamma \mathbb{E}_{a'\sim \pi(·|s')}Q(s', a')\big]$，$Q(s',a')$ 表示在下一个状态 $s'$ 下，动作 $a'$ 的价值
- **最优动作价值函数的贝尔曼方程**：$Q(s,a)=\mathbb{E}_{s',r\sim P}(r(s,a,s')+\gamma \max\limits_{a'}Q(s', a'))$

#### Model-Based (动态规划)

**需要知道 $r(s, a, s')$ 和 $P(s'|s, a)$**

核心思想：将全局的优化问题拆分为多个子问题求解，这也是动态规划的精髓

**贝尔曼方程**（用于解构问题）：

$$
\begin{aligned}
V(s)
&= \mathbb{E}_{a\sim \pi(·|s)}\mathbb{E}_{s'\sim P(s'|s, a)}\big[r(s, a, s')+\gamma V(s')\big] \\
&= \sum\limits_{a}\pi(a|s)\sum\limits_{s'}P(s'|s,a)\big(r(s,a,s')+\gamma V(s')\big)
\end{aligned}
\hspace{0pt}
$$

- 策略迭代：
  - **评估** 先固定策略 $\pi$，用贝尔曼方程迭代更新值函数：
    - $V_{k+1}(s) = \sum\limits_{a}\pi(a|s)\sum\limits_{s'}P(s'|s,a)\big(r(s,a,s')+\gamma V_k(s')\big)$
    - $Q_{k+1}(s,a) = \sum\limits_{s'}P(s'|s,a)(r(s,a,s')+\gamma \sum\limits_{a'}\pi(a'|s')\cdot Q_k(s',a'))$
  - **优化** 策略提升的方法：
    -  $\pi'(s) = \argmax\limits_{a}Q_{k+1}(s,a) = \argmax\limits_{a}\sum\limits_{s'}P(s'|s,a)(r(s,a,s')+\gamma V_k(s'))$
  - 保证 $V_{k+1}(s)\ge V_k(s)$，且当 $\max|V_{k+1}(s)-V_k(s)|\le \epsilon$ 时算法收敛
- 价值迭代：
  - **评估+优化** 一边迭代，一边更新策略：
    - $V_{k+1}(s) = \max\limits_{a\sim \pi(·|s)}\sum\limits_{s'}P(s'|s,a)(r(s,a,s')+\gamma V_k(s'))$
    - $Q_{k+1}(s,a) = \sum\limits_{s'}P(s'|s,a)(r(s,a,s')+\gamma\max\limits_{a'}Q_k(s',a'))$
  - 收敛后直接得到最终的策略：
    - $\pi(s) = \argmax\limits_{a}Q(s,a) = \argmax\limits_{a}\sum\limits_{s'}P(s'|s,a)(r(s,a,s')+\gamma V(s'))$

#### Model-Free

无需知道环境的模型 $P(s'|s,a)$，只需要设计奖励模型 $r(s,a,s')$，更新的是动作价值函数 $Q(s,a)$

主要有三类方法：

- **蒙特卡洛**（基于回合的迭代方法，必须要完成一个回合后才能更新）
- **时序差分**

蒙特卡洛方法：

- 使用策略 $\pi$ 从状态 $s_t$ 开始，完成一回合完整的交互，得到一组奖励 $\{R_{+1}, R_{t+2}, ..., R_T \}$， $T$ 为回合终止时的状态，定义折扣回报 $G_t = R_{t+1}+\gamma R_{t+2}+...+ \gamma^{T-t-1}R_T$
- 定义状态价值函数：
  - $V_{\pi}(s)=\mathbb{E}_{\pi}[G_t|S=s_t]$，用期望近似真值是 MC 的思想
- 实际工程实现使用增量更新公式：
  - $V_{\pi}(s_t) \leftarrow V_{\pi}(s_t) + \alpha(G_t-V_{\pi}(s_t))$
  - 每次都采样一条完整的轨迹，计算 $G_t$，更新 $V_{\pi}(s_t)$ 直至收敛

时序差分方法:

- **SARSA** (state-action-reward-state-action)
  - 同策略方法（**On-Policy**）
  - 先从当前策略中采样一个五元组 $(s,a,r,s',a')$
  - 使用 $Q(s,a) \leftarrow Q(s,a)+\alpha(r(s,a,s') + \gamma Q(s',a')-Q(s,a))$ 更新动作价值函数
  - 这里的 $a'$ 是策略 **$\pi$ 实际会选择的动作** (常用 $\epsilon-greedy$，小概率随机，大概率经验)
- **Q-Learning**
  - 异策略方法（**Off-Policy**）
  - 采样策略 $\not =$ 目标策略，数据可以重复使用，**无需策略的决策**就能更新价值函数
  - $Q(s,a) \leftarrow Q(s,a)+\alpha(r(s,a,s') + \gamma\max\limits_{a'} Q(s',a')-Q(s,a))$
  - 没有用到 $a'$，只是对状态 $s'$ 下的所有动作值取最大
- **DQN** (Deep Q-Network)
  - 异策略方法（**Off-Policy**）
  - Q-Learning 维护一张 Q 表，记录状态动作对的价值，内存开销大，动作维度过高会引发维度灾难
  - DQN 使用神经网络 $Q(s,a;\theta)$ 近似动作价值函数，输入状态 $s$，输出每个动作 $a$ 的 $Q$ 值
  - 采样策略：
    - 基本训练数据：五元组 $(s,a,r,s',done)$，$done$ 是一个布尔值，表示轨迹是否终止
    - 使用 $\epsilon-greedy$ 策略，以小概率做随机动作，否则 $a=\argmax\limits_{a}Q(s,a;\theta)$
    - 指定动作得到 $s', r(s,a,s')$
    - 记录五元组 $(s,a,r,s',done)$，存放到 D 中
  - 训练策略：
    - 由于同一条轨迹上的样本强相关，用这样的数据更新模型训练非常不稳定（方差大），于是使用一个**回放缓冲区 D**，存放采样得到的样本（有旧的也有新的），训练时从其中随机采样，可以得到近似独立同分布的样本
    - 为了保证训练的稳定性，同时训练新旧两个网络：新网络称为**主网络($Q(;\theta)$)**，用每个样本得到的损失进行优化，旧网络称为**目标网络($\hat{Q}(\hat{\theta})$)**，为待优化的网络，主网络每更新 $C$ 次就会覆盖旧网络($\hat{\theta}\leftarrow\theta$)，使得目标网络的优化过程循序渐进
    - 采用奖励裁剪策略，防止梯度过大，也是为了保证训练的稳定性
  - 更新策略：
    - 取样本 $x_i=(s_i,a_i,r_i,s_i',done)$
    - if done==true: $y_i=r_i$
    - else: $y_i=r_i+\gamma\argmax\limits_{a'}\hat{Q}(s_i',a';\hat{\theta})$
    - 计算 $L(\theta)=\mathbb{E}[(y_i-Q(s_i,a_i;\theta))^2]$，对批量样本做均值
    - 求梯度后反传**优化 $\theta$**，更新 $C$ 步后执行 $\hat{\theta}\leftarrow\theta$

同策略方法更保守，更稳定，但更新速度缓慢；异策略方法更激进，更新速度快，并且数据可以重复使用，但容易高估动作价值

### 基于策略梯度的强化学习方法

基于值函数的强化学习方法使用值函数（$V(s)$ 或 $Q(s,a)$），间接地得到策略 $\pi(s)$

基于策略梯度的强化学习方法与此不同，直接将策略 $\pi(a|s)$ 进行**参数化**，它们的区别如下

| 对比项      | 值函数方法     | 策略梯度方法|
|-------------|-----------------------|------------------------|
| 学习对象    | Q / V 值               | 策略概率分布            |
| 动作空间    | 适合**离散**，不适合连续 | 天然适合**连续**，也支持离散 |
| 策略类型    | 近似确定性              | 随机策略               |
| 样本效率    | 高（离线、经验回放）     | 低（大多在线）         |
| 训练稳定性  | 稳定                   | 波动大、方差高          |
| 收敛性      | 较好                   | 易局部最优             |
| 核心问题    | Q 过估计、连续动作难处理 | 方差大、样本低效       |
| 典型算法    | SARSA、Q-Learning、DQN | REINFORCE、A2C、PPO    |

#### REINFORCE (On-Policy)

最纯朴的**蒙特卡洛算法**

$\pi_{\theta}(a|s)$ 为待优化的策略，目标是最大化累计收益：

$$J(\theta)=\mathbb{E}_{r\sim \pi_{\theta}}\big[\sum\limits_{\tau\sim \pi_{\theta}}r(s_t,a_t)\big]$$

其中 $\tau$ 表示整条轨迹 $\{(s_0,a_0),(s_1,a_1),...,(s_{T},done)\}$

**策略梯度**的推导过程如下：

记 $R(\tau)=\sum\limits_{\tau}r(s_t,a_t)$，采样到轨迹 $\tau$ 的概率为 $p_{\theta}(\tau)$，于是累计收益可表示为

$$J(\theta)=\int p_{\theta}(\tau)\cdot R(\tau)d\tau$$

$$\nabla_{\theta} J(\theta)=\int \nabla_{\theta}p_{\theta}(\tau)\cdot R(\tau)d\tau$$

由于 $\nabla \log p_{\theta}=\frac{\nabla_{\theta}p_{\theta}}{p_{\theta}}$，所以可知 $\nabla_{\theta} p_{\theta}(\tau)=p_{\theta}(\tau)\cdot\nabla_{\theta}\log p_{\theta}(\tau)$，于是有：

$$\nabla_{\theta} J(\theta)=\int p_{\theta}(\tau)\cdot\nabla_{\theta}\log p_{\theta}(\tau)\cdot R(\tau)d\tau=\mathbb{E}_{\tau\sim\pi_{\theta}}\big[\nabla_{\theta}\log p_{\theta}(\tau)\cdot R(\tau)\big]$$

由概率乘法公式可知 $ p_{\theta}(\tau)=p(s_0)\cdot\prod\limits_{t=0}^{T-1}\pi_{\theta}(a_t|s_t)\cdot p(s_{t+1}|s_t,a_t) $，$p(s_{t+1}|s_t,a_t)$ 为**环境的状态转移概率**，于是 $ \log p_{\theta}(\tau)=\log p(s_0)+\sum\limits_{t}\log\pi_{\theta}(a_t|s_t)+\sum\limits_{t}\log p(s_{t+1}|s_t,a_t) $

所以 $\nabla_{\theta} \log p_{\theta}(\tau)=\sum\limits_{t}\nabla_{\theta}\log\pi_{\theta}(a_t|s_t) $，于是有：

$$\nabla_{\theta}J(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta}}\big[R(\tau)\cdot \sum\limits_{t}\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\big]$$

实际工程中并不可能遍历所有的轨迹 $\tau$ 求得真实的期望，只能采用小批量样本计算梯度来近似真实值，工程上会采用的方法是用采样到的**单个样本**直接更新参数：

- 用当前策略 $\pi_{\theta}$ 采样一条或多条完整的轨迹 $\tau$
- 计算每条轨迹的累计收益 $R(\tau)$
- 由于是最大化收益函数，采用梯度上升：$\theta \leftarrow \theta+\alpha\cdot\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\cdot R(\tau)$

缺点：由于 $R(\tau)$ 涉及多个随机过程，其**方差很大**，导致梯度的方差也较大

**控制梯度的方差是保证训练稳定性的重要因素**

#### Actor-Critic (On-Policy)

不去直接采样整条轨迹，使用 **Critic** 网络估计某个状态的平均收益 $V(s)$，**Actor** 网络输出某个状态下的动作概率分布

$V(s)$ 的估计方法和基于值函数的强化学习方法中类似，可采用时序差分进行优化：

$$L_V=\mathbb{E}\big[\big(V(s_t)-(r_t+\gamma V(s_{t+1}))\big)^2\big]$$

求 $L_V$ 的梯度，最小化这个 MSE，更新 Critic 网络 V(s)，进而可以估计动作价值：

$$Q(s,a)=\mathbb{E}\big[r(s,a,s')+\gamma V(s')\big]$$

s' 表示 s 采取动作 a 后转移到的状态，单样本估计情况下 $Q(s,a)=r(s,a,s')+\gamma V(s')$

用**优势函数**代替蒙特卡洛算法中的累计收益函数，可以减小梯度的方差：

$$A(s,a)=Q(s,a)-V(s)=r(s,a,s')+\gamma V(s')-V(s)$$

$$\nabla_{\theta}J(\theta)=\mathbb{E}\big[\sum\limits_{t}\nabla_{\theta}\pi_{\theta}(a_t|s_t)\cdot A(s_t,a_t)\big]$$

此外还使用**梯度裁剪**的方法降低梯度估计的方差

- 初始化 Critic、Actor 网络与优化器
- 环境交互：$s_t\rightarrow$ Actor $\rightarrow a_t$，$a_t\rightarrow$ Critic $\rightarrow s_{t+1},r_t$
- Critic 计算 $L_V$，估计优势 $A(s_t,a_t)$
- 经过梯度裁剪后，更新 Critic（梯度下降） 和 Actor（梯度上升）

虽然 $A(s,a)$ 只引入了一步随机性，方差较小，并且引入了 **Baseline** V(s) 和梯度裁剪策略，但是估计的精度取决于 $V(s)$，其**偏差可能较大**

#### PPO (Off-Policy)

基于 Actor-Ctiric 架构优化的强化学习算法，主要解决以下两个问题：

- 训练不稳定，估计的策略梯度**偏差和方差**较大
- on-policy 方法，**采样成本较高**

PPO 算法用一组参数 $\theta$ 同时表示 Actor 和 Critic，二者共享一部分参数，通过输出头区分

对于采样效率问题，PPO 算法通过**重要性采样**将其转化为 **off-policy** 问题

重要性采样：

$$\mathbb{E}_{x\sim p(x)}\big[ f(x) \big]=\int p(x)\cdot f(x)dx=\int q(x)\cdot\frac{p(x)}{q(x)}f(x)dx=\mathbb{E}_{x\sim q(x)}\big[\frac{p(x)}{q(x)}\cdot f(x) \big]$$

策略梯度的估计:

- $\nabla_{\theta}J(\theta)=\mathbb{E}_{\theta}\big[\sum\limits_{t} \nabla_{\theta}\log \pi_{\theta}(a_t|s_t)\cdot A_t^{GAE} \big]=\mathbb{E}_{\theta_{old}}\big[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\sum\limits_{t} \nabla_{\theta}\log \pi_{\theta}(a_t|s_t)\cdot A_t^{GAE} \big]$

对于策略梯度的方差和偏差问题，PPO 算法通过使用**折扣回报**、**广义优势估计**，以及**损失函数裁剪**解决：

- 折扣回报：$R(\tau)=\sum\limits_{t}\gamma^{t-1}\cdot r(s_t, a_t)$
- 广义优势估计：
  - TD(时序差分)误差：$\delta_t=r(s_t,a_t,s_{t+1})+\gamma V(s_{t+1})-V(s_t)$
  - n步优势函数估计：$A_t^{(n)}=\sum\limits_{l=0}^{n-1}\gamma^{l}\delta_{t+l}$
  - 广义优势估计（将不同n步估计做指数加权）：$A_t^{GAE}=(1-\lambda)\sum\limits_{i=1}^{\infty}\lambda^{i-1}A_t^{(i)}$
  - 通过交换求和顺序、化简指数求和，可得 $A_t^{GAE}=\sum\limits_{l=0}^\infty (\gamma\lambda)^l\delta_{t+l}$
  - 编码中常用形式 $A_t^{GAE}=\delta_t+\gamma\lambda\cdot A_{t+1}^{GAE}$
- 损失函数裁剪：
  - PPO的目损失函数 $L(\theta)=\mathbb{E}_t\big[\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_old}(a_t|s_t)}\cdot \nabla \log\pi_{\theta}(a_t|s_t)\cdot A_t^{GAE}\big]$
  - 将目标函数限制在 1 的附近 $L^{CLIP}=clip(L(\theta), 1-\epsilon, 1+\epsilon)$

#### GRPO (On-Policy)
