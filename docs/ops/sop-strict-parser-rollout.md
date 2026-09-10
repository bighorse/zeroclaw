# SOP 步骤解析器推广（strict）—— 全量排查结果与待办

日期：2026-09-11。排查覆盖 clawops（120.48.131.72）与 ai.infocts.cn（47.94.58.57）
两台生产机上的全部 SOP：**19 份互不相同的 SOP.md，分布在 34 个副本中，外加 96 个
逐实例个性化过的 policy-match**。每份由一个独立 agent 读原文裁定作者意图，再由
另一个 agent 对抗性复核（38 个 agent，2 条裁定被推翻并已更正）。

复核工具：`cargo run --example sop_parse_diff -- <SOP 目录>…`（加 `--json` 出机器可读）。

## 一、缺陷是什么

默认（legacy）解析器有三种静默失效：

1. **缩进的编号子项被当成步骤**。`- notes:` 下面的 `1. / 2. …` 全部变成"步骤"，
   而每认一个新步骤就会重置 `requires_confirmation`，人工审批门随之丢失。
2. **步骤正文里的 `## 标题` 提前终止解析**。步骤里贴的产物模板（哪怕在
   ``` 围栏内）含 `## 1. 病例摘要` 就会被当成"另起一节"，其后的步骤连同人工门直接不存在。
3. **`### N. 标题` 式的步骤完全不被识别**。生产上 `kf-followup` 与 `semester-report`
   都是这么写的。

第 3 种最隐蔽：门不是"丢了"，而是**挂错了地方**。kf-followup 的
`requires_confirmation` 落在一条叫「重试三次仍失败则终止」的 notes 子项上——
`sop status` 看上去有门，实际那道门守着一个不存在的关口。

## 二、已做的修复（fork 分支 `feat/frontdesk-v6-p0`）

- strict 现在把 Steps 段内**顶格**的 `#{2,6} <数字>[.、] <标题>` 当作步骤。
  缩进的同形标题仍算正文——产物模板里 `### 1. 病例摘要` 若被提为步骤，
  就是 legacy 那个缺陷的镜像。kf-followup 由 0 步变为其意图的 6 步，
  semester-report 由 0 步变为 7 步且人工门在第 7 步。**legacy 行为未动。**
- 加载未声明 `step_parser` 的 SOP 时，两种解析结果不一致就 `warn!`，
  并对"文本写了人工门、默认解析器一个都没解析出来"单独重话点名。**只记日志，不改行为。**
  没有这条，以后每写一个新 SOP 都要靠人再扫一遍。

## 三、已切到 strict 的 SOP（逐条复核过）

| SOP | 意图步数 | 人工门 | 位置 |
|---|---|---|---|
| case-clinical-report | 5 | 4 | zeroclaw-builder/产物/工作流 |
| cbp-data-refresh | 6 | 5 | 同上 |
| lead-to-match | 4 | 3 | 同上 |
| batch-characterization | 7 | 1、6 | zeroclaw-builder/示例产物、zeroclaw-workbench-deploy/builder |
| jl-transomics-intake | 4 | 2 | zeroclaw/deployments/geneline |
| jl-insight-research-proposal | 15 | 9 | 同上 |

## 四、仍需 SOP 负责人改 SOP.md 的（解析器修不了）

这几份的正文明确要求人工把关，但**没有任何一步写了顶层
`- requires_confirmation: true`**。这不是"strict 看不见"，是根本没写——
换哪个解析器都不会凭 Quality Gates 段的散文造出一道门。

| SOP | 缺的门 | 在跑的实例 | 依据 |
|---|---|---|---|
| enterprise-quick-review 1.1.0 | 第 1 步 | **intel-claw** | 正文「用户确认后再继续」 |
| enterprise-quick-review 1.4.0 | 第 1、2 步 | — | 同源 |
| kf-followup 1.3.0 | 第 3 步 | **loopclaw-frontdesk** | 发出前须人工确认 |
| semester-report 1.0.0 | 第 1 步 | **shuizhiya** | 「【P0 强制：学期确认】先跟用户确认学期标识」（第 7 步的门已由本次解析器修复补回） |
| investment-monthly-workflow 1.1.0 | 第 1 步 | **ai-staff** | 「缺时必须先向用户确认，不得凭空假设」+ Quality Gate「引擎必须拒绝推进」；当前 `- requires_confirmation: false` 是显式写死的 |
| lead-to-match（clawops 产物副本） | 第 4 步 | — | — |

**四份正在生产上跑，且这四份都不是"推 strict"能解决的。**

## 五、不建议动的

- **policy-match（96 个实例）**：96 份 SOP.md 内容各不相同（逐客户个性化），
  但结构完全一致：legacy 8 步、strict 6 步，**两边都没有人工门**。
  没有合规风险，为 2 个虚步骤重启 96 个实例不划算。等各实例下次自然重启再说。
- **em-phase-transition 2.0.0（emclaw）/ visitor-escalation**：两种解析结果一致，无需改。

## 六、切换的操作代价（比预想小）

各裁定 agent 都担心「步号重编导致在飞任务跳门」。**这个风险实际不存在**：
SOP 只在进程启动时加载（`src/daemon/mod.rs:61`、`src/gateway/mod.rs:453`，无热加载），
而 run 只存在内存里，重启即丢。也就是说，改 SOP.toml 必须重启，而重启本身
已经清空了所有 `current_step`。代价只是「重启一次 + 在飞任务作废」，
与平时升级二进制同级。

唯一要留意的是**切换后流程会第一次真的停下来等人**（此前是一路裸奔到底）：
切之前要确认审批通道确实有人在、通得了，否则从"无门裸奔"变成"有门但没人来开"。
