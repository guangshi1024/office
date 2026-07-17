# LiteAgent（轻量智能体）

> [!warning]
> Simple Agent 可以在你的机器上运行命令。请注意它可能会对你的系统进行修改。

![Aug-16-2024 11-27-09](https://github.com/user-attachments/assets/3bb43a56-0501-4759-b3b0-ac459f53f692)

> [!note]
> Simple Agent 仍在开发中！可能会存在 Bug 和缺失的功能，如果你遇到了这些问题，请提交一个 issue！

这是一个基于 LLM 和其他技术构建极简、简单智能体的尝试。目标是拥有一个可用于多种用途的简单智能体：

1. 实验与基准测试
2. 自我理解与学习
3. 我也不知道，但凑三个点听起来更靠谱

## 工作原理

简单来说，该智能体在一个持续的循环中运行，包含三个主要步骤：

1. 感知（Perception）
2. 推理（Inference）
3. 行动（Action）

感知阶段的特点是将以下三者结合起来：1. 环境分析，2. 记忆召回，以及 3. 面向目标的主动性或意志。推理阶段是调用 API，将感知结果交给模型（目前仅支持 OpenAI 模型）进行推理。行动阶段是智能体获取模型的输出（如工具调用）并实际执行它们。

## 前置条件
要让 Simple Agent 良好运行，你需要满足以下前置条件：
- Python 3.11
- 能够访问 LLM（目前默认仅支持 Anthropic 或 OpenAI）

## 如何使用

这个仓库是一个相当直接的 Python 项目。**你需要安装 Python**，我在 MacOS 上使用的是 Python 3.10，所以我知道这个版本是可以用的。如果你使用不同版本，或在 MacOS 以外的平台上使用并发现了 Bug，请在 issue 中告诉我。你应该能够克隆仓库，安装依赖（如果需要，可以在虚拟环境中安装），填写 `.env` 文件，然后运行 `simple-agent.py` 来查看效果。

1. 首先克隆仓库（如果想大量扩展，可以先 Fork）：

```bash
   git clone https://github.com/guangshi1024/LiteAgent.git
```

2. 然后，进入目录：

```bash
cd LiteAgent
```

3a.（可选）如果你选择使用虚拟环境，可以这样创建：

```bash
python -m venv venv
```

然后激活虚拟环境：

```bash
source ./venv/bin/activate
```

3b. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 将 `.env.example` 文件复制为 `.env` 并填写必要的信息。你需要一个 OpenAI API 密钥，可以从 [OpenAI 网站](https://platform.openai.com/api-keys) 获取。

（在 MacOS 或 Linux 上）：

```bash
cp .env.example .env
```

5. 运行程序：

```bash
python simple-agent.py
```

6. 要退出程序，可以使用 `Ctrl+C`，或者输入 "exit"。

就这样！你应该能看到智能体启动，并且你可以让它使用工具来做各种事情。

## 高级用法

你可能想扩展或增强该工具的功能，如果你这样做了，我很乐意听听效果如何！如果你认为某个功能应该成为默认功能，甚至可以提交 PR。

**修改提示词**
你可以在 `agent.py`、`agency.py`、`memory.py` 和 `environment.py` 文件中修改提示词。这些文件包含了提示词中多个方面的实际程序化生成逻辑。随着时间的推移，这些内容可能会发生变化，所以我不会在这里详细展开。但你可能会发现，通过修改这些文件可以获得更好的行为表现。

`system_prompt.md` 文件是智能体系统提示词的来源，同样可以进行修改。

**使用不同的模型**
你可能会发现使用不同的模型更有用。然而，每个服务提供商暴露的模型接口略有不同，这使得在某些情况下使用它们并不是一个简单的过程。尽管如此，`LLM` 类允许将 Simple Agent 的标准数据结构轻松适配到不同的 API。目前，默认支持 OpenAI、Anthropic 和 Gemini API，对应文件位于 `llms/` 目录中。你可以通过 `MODEL_CHOICE` 设置要使用的提供商，然后在 `.env` 中设置相应的 `[PROVIDER]_MODEL` 和 `[PROVIDER]_API_KEY` 环境变量来配置具体的模型。

**添加工具**
`tools/` 目录是你可以找到工具的地方。每个工具必须符合 `Tool` 类才能被智能体使用。查看 `toolbox.py` 可以了解当前包含的工具列表以及它们的使用方式。如果你想添加工具，可以参考 `write_file.py`、`read_file.py` 和 `send_message_to_user.py` 这些工具。未来会加入更多工具。这也是欢迎 PR 的另一个领域。

## 角色（Roles）
角色是智能体可以采纳的身份。它们是上下文相关的，每个角色都有自己的一套工具提供给智能体。智能体可以随时在角色之间切换。`INCLUDED_ROLES` 在 `roles/config.py` 文件中定义。你可以在[角色教程](/documentation/adding-a-role.md)中了解更多关于角色的信息。

我仍在完善角色系统，请耐心等待。但你可以在 `roles/` 目录中看到它的雏形。目前支持以下角色：
- `Helpful Assistant`（乐于助人的助手）：一个乐于助人的助手，随时准备接收你的问题并提供答案。
- `Developer`（开发者）：一个专注的开发者，拥有用于软件开发、测试和调试的专业工具。
- `Researcher`（研究员）：一个专注的研究员，拥有用于网络研究、数据分析和文档编写的专业工具。

> [!note]
> 要切换角色，只需让智能体去做即可。例如，你可以说"我想要一些研究方面的帮助……"，或者更具体地说"我想切换到研究员角色"。你会看到一条 "As a [ROLE]" 的消息来确认。

## 添加记忆
查看[此教程](/documentation/adding-memory.md)了解如何为 Simple Agent 添加记忆。这并非智能体运行所必需，但可以让其记忆在会话之间持久化。

## 路线图
我希望将这个项目打磨成一个真正扎实的智能体构建框架。但它需要保持为核心框架，而非一个功能完备的智能体。要构建功能完备的智能体，应该基于此项目进行 Fork。就目前的状态而言，它是可用、功能齐全且可扩展的。

## 功能
**已有功能**
- [x] 通过 [Simple Vector Store](https://github.com/AidanTilgner/Simple-Vector-Store) 实现记忆和学习
- [x] 允许选择更多模型
- [x] 获得更好的编辑体验
- [x] 智能体可以切换进入的"模式"，用于专业化操作，拥有不同的工具布局

**进行中（大概）的功能**
这些是我目前计划做或希望做的事情。如果你有功能想法或反馈，请随时提交 issue 以便我们讨论。
- [ ] 通过 [Benchy](https://github.com/AidanTilgner/Benchy) 实现更健壮的基准测试系统
- [ ] 一种程序性记忆
- [ ] 一种插件系统
