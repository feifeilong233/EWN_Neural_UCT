# EWN-Neural：基于神经网络增强 UCT 的爱因斯坦棋博弈程序

> **项目全称**：Einstein Würfelt Nicht! — Neural-Enhanced UCT Algorithm
>
> **用途**：2024 年 ICGA 计算机博弈大赛参赛程序 / 学术论文实验平台
>
> **开发者**：温飞龙 (Feilong Wen / yakumohakuryu / wenfeilong233)
>
> **开发周期**：2024 年 3 月 — 2025 年 3 月

---

## 1. 项目概述

本项目实现了**爱因斯坦棋（Einstein Würfelt Nicht!）** 的计算机博弈智能体。核心算法为**蒙特卡洛树搜索（MCTS）的 UCT 变体**，并利用 **LibTorch（PyTorch C++ 前端）** 加载深度神经网络进行局面评估与走子概率预测，从而显著提升搜索效率和棋力。程序通过 TCP Socket 与外部裁判/对战服务器通信，遵循客户端-服务器博弈协议。

### 1.1 棋种简介

爱因斯坦棋是一种双人骰子驱动的不完全信息棋类游戏。双方各有 6 枚棋子（红方编号 1–6，蓝方编号 7–12），位于 5×5 棋盘的对角两端。每回合掷骰子决定可移动的棋子，红方向右下方向移动，蓝方向左上方向移动。胜利条件为：率先将己方全部棋子移至对方底线，或消灭对方全部棋子。

### 1.2 算法特色

| 模块 | 实现方式 |
|------|----------|
| **树搜索策略** | UCT（Upper Confidence Bound applied to Trees） |
| **估值函数** | 基于概率期望的 EVS（Expected Value Score），含威胁值评估 |
| **神经网络** | LibTorch 加载 PyTorch 模型（`Charlie2.pt` / `Alpha1.pt`），输出走子策略概率 |
| **模拟策略** | 网络引导的带温度 Softmax 随机选择（`NeuralSimulate`） |
| **残局库** | QNNTS 分支引入，加速终局搜索（`Endgame.h`） |
| **并行计算** | OpenMP 多线程加速 |
| **GPU 加速** | 支持 CUDA 设备推理（`device_type = at::kCUDA`） |

### 1.3 技术栈

- **语言**：C++14
- **神经网络**：LibTorch（PyTorch 1.x C++ API）
- **并行**：OpenMP
- **构建**：CMake ≥ 3.12
- **通信**：Windows Socket API（Winsock2）
- **IDE**：CLion (JetBrains)

---

## 2. 项目结构

```
EWN_Neural/
├── main.cpp            # 程序入口，创建 CEinstein 实例并循环 handle()
├── CEinstein.h/cpp     # 核心类：通信解析、UCT 搜索（~1550 行）
├── CFakeBoard.h        # 轻量伪棋盘类，用于快速局面估值
├── ClientSocket.h/cpp  # TCP Socket 通信层（与裁判服务器交互）
├── Define.h            # 全局宏定义（ID、服务器 IP/端口）
├── CMakeLists.txt      # CMake 构建配置（LibTorch + OpenMP）
├── Charlie2.pt         # 网络模型权重文件（当前 HEAD 使用）
├── Alpha1.pt           # 网络模型权重文件（QNNTS 分支使用，约 19.7 MB）
├── Endgame.h           # 残局库（QNNTS 分支专属）
└── README.md           # 本文件
```

---

## 3. Git 分支管理与演进历程

本项目采用多分支并行开发策略，各分支服务于不同的实验与比赛目的。以下按时间线梳理分支关系与功能定位。

### 3.1 分支总览

```
                    f71e5cf (2024-03-28)  Initial commit
                        │
                        ├── master ───────────────────── fd151c5 (2024-05-25)
                        │   └─ 基础 UCT + 网络参与选择阶段（实验原型）
                        │
                        ├── N-UCT ───────────────────── 12fa3e8 (2024-07-02)
                        │   └─ **论文实验主分支**
                        │      · 棋盘视角对称变换（蓝方视为红方）
                        │      · 历史状态队列 (status deque)
                        │      · 开局条件检测
                        │      · 随机数生成器修正
                        │
                        ├── RCNSS-UCT ───────────────── bc7c769 (2024-12-08)
                        │   └─ RC+NSS 融合实验分支
                        │      · 蓝方随机策略
                        │      · 选择阶段网络集成
                        │      · 服务器线上测试
                        │
                        └── QNNTS (origin/QNNTS) ────── c6ff96f (2025-03-17)
                            └─ **ICGA 2024 大赛主程序**
                               · 完整神经网络引导模拟
                               · 残局库 (Endgame.h)
                               · GUI 集成
                               · 红方开局修复
                               · 时间种子随机
                               · 赛后持续优化
```

### 3.2 分支详细说明

#### master（基础原型）
- **生命周期**：2024-03-28 — 2024-05-25
- **定位**：项目起点，验证基本框架可行性
- **关键提交**：
  - `f71e5cf` — 初始提交，搭建 C/S 通信和基础 UCT 骨架
  - `510a6f3` — 改为单步 15 秒时限、约 1000 次模拟
  - `fd151c5` — 网络加入选择阶段（Selection Phase）的初步尝试
- **说明**：此分支为最简化版本，仅用于最初的原型验证和工作站实验

#### N-UCT（论文实验主分支）
- **生命周期**：2024-03-28 — 2024-07-02
- **定位**：学术论文实验与数据采集的核心分支（"Neural-UCT"）
- **关键提交**：
  - `a3ba543` — **蓝方视为红方**：对棋盘做视角对称变换（`board[24-i]`），统一红方推理逻辑。此为算法的关键创新点
  - `c00d157` — 维护一个局面的**历史状态队列**（`std::deque<array<array<int,5>,5>> status`），为网络提供时序特征
  - `7e205a0` — 实现**开局条件判断**（`checkOpeningCondition`），针对空棋盘扩展初节点
  - `4518503` — 修复随机数生成器实现
- **说明**：此分支的算法设计直接支撑了后续 QNNTS 的实现，是论文方法的核心代码

#### RCNSS-UCT（融合实验分支）
- **生命周期**：派生自 QNNTS 早期阶段，持续至 2024-12-08
- **定位**：RC（Random-Choice/Reinforcement Component）与 NSS（Neural State Scoring）融合实验
- **关键提交**：
  - `1cf6a42` — "彻底仿照 QNNTS 算法，目前还差揉进 RC"
  - `72927db` — RC 融入蓝方，红方代码整合
  - `ca64be3` — 发现扩展初节点无访问次数问题（研究记录）
  - `ad53568` — 蓝方改为随机策略，标记为"就这吧"
  - `bc7c769` — 部署至服务器测试
- **说明**：此分支为探索性实验，测试了多种策略组合后终止，其部分思想可能被吸收至主分支

#### QNNTS（ICGA 2024 大赛主程序）
- **生命周期**：2024-08-16 — 2025-03-17
- **定位**：**ICGA 2024 计算机博弈大赛参赛版本**，也是项目最终的完整版本
- **比赛阶段（2024-08-23 — 2024-08-28）**：
  - `c7efaa2` — "QNNTS の完全体"：完整算法框架就绪
  - `2570288` — 红方开局判断修复，随机种子改为系统时间
  - `baf13d3` — 接入 GUI 开始密集测试
  - `84bfe0d` — **最终比赛配置：15 秒时限、300 次模拟**
  - `edb694c` — "结束了"：比赛圆满结束
- **赛后增强（2024-12 — 2025-03）**：
  - `b9ae986` — QNNTS 测试框架搭建
  - `9fa9833` — 准备加入残局库
  - `7b596a0` — **残局库（Endgame.h）** 正式引入
  - `c6ff96f` — "NPure"：最终精炼版本
- **说明**：QNNTS 在 N-UCT 基础上引入了神经网络引导的**完整模拟阶段**（而非仅在 Selection 使用）、残局库加速、以及 EVS 估值驱动的蓝方决策，棋力得到显著提升

### 3.3 分支演化关系图

```
 [Initial] ───> master ───> N-UCT (论文实验)
                  │              │
                  │              ├──> RCNSS-UCT (融合实验，派生自 QNNTS 早期)
                  │              │
                  └──> QNNTS ───┘    (ICGA 大赛)
                          │
                          ├── 2024-08 比赛密集开发
                          ├── 2024-12 残局库 + 测试框架
                          └── 2025-03 NPure 精炼版
```

### 3.4 当前 HEAD 状态

当前工作目录处于 `7be4448`（detached HEAD），位于 QNNTS 分支的起点附近（2024-08-24），是 N-UCT 与 QNNTS 的分叉点之前的一个提交。最新稳定版请查阅对应分支。

---

## 4. 编译与运行

### 4.1 依赖

- **Windows 环境**（依赖 Winsock2）
- **Visual Studio 2019/2022** 或 MinGW-w64
- **CMake** ≥ 3.12
- **LibTorch**（C++ 分发包，需与 PyTorch 版本匹配）
- **OpenMP**（通常随编译器附带）

### 4.2 构建

在 `CMakeLists.txt` 中修改 `Torch_DIR` 指向你的 LibTorch 安装路径：

```cmake
set(Torch_DIR "C:/libtorch/share/cmake/Torch")
```

然后执行：

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 4.3 运行

程序通过 TCP Socket 连接裁判服务器。在 `Define.h` 中配置：

```cpp
#define ID          "Neural-UCT"    // 选手标识
#define SERVER_IP   "127.0.0.1"     // 裁判服务器地址
#define SERVER_PORT 50007            // 裁判服务器端口
```

将编译好的网络模型文件（`.pt`）放在可执行文件同目录下，启动裁判服务器后运行程序即可自动对战。

---

## 5. 关键算法参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| UCB 探索系数 | C | 0.85 | 控制探索-利用平衡 |
| EVS 加成系数 | k | 1.15 | 估值函数在 UCB 中的权重 |
| 估值 λ | λ | 4.98 | EVS 得分的缩放系数 |
| 估值 β | β | 2.2 | 红方期望的加权系数 |
| 网络先验 α | α | 9 | 网络概率在 Selection 中的权重 |
| 模拟温度 τ | tempo | 0.1 | 神经网络走子 Softmax 温度 |
| 蓝方温度 | tempo_demo | 2.0 | 蓝方 EVS 选择温度 |
| 先验访问基数 | C_B | 300 | 网络先验的虚拟访问次数 |
| 思考时限 | — | 5 秒 | 单步搜索时间（ICGA 比赛为 15 秒） |

---

## 6. 研究成果与输出

- **ICGA 2024 计算机博弈大赛**：以 QNNTS 分支程序参赛
- **学术论文**：以 N-UCT 分支为核心实验代码，研究成果涵盖：
  - 棋盘视角对称变换统一推理逻辑
  - 历史状态序列的时序特征编码
  - 神经网络引导的 UCT 搜索优化

---

## 7. 参考文档

- 爱因斯坦棋规则与策略：[Einstein Würfelt Nicht! on BoardGameGeek](https://boardgamegeek.com/boardgame/23808/einstein-wurfelt-nicht)
- MCTS/UCT 算法：[UCT 算法详解 (cnblogs)](https://www.cnblogs.com/KakagouLT/p/9201845.html)
- LibTorch C++ 教程：[PyTorch C++ API](https://pytorch.org/cppdocs/)

---

## 8. 致谢

- 感谢 ICGA 2024 组委会提供的比赛平台与评价体系
- 感谢导师在论文研究与实验设计中的悉心指导
- 感谢实验室服务器提供的计算资源支持

---

*最后更新：2026 年 6 月 27 日 | 毕业归档版本*
