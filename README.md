# SoulDrive

SoulDrive 是一个面向私有资产的端侧知识引擎，核心目标是把本地知识文档的解析、索引、检索增强生成、证据链和知识图谱打磨稳定。当前演示与测试仍以论文作为默认切入口，但系统能力不局限于论文场景。

## 核心模块

- `core/`：本地 sidecar、工作区管理、文档解析、索引、检索、RAG、审计日志、模型运行配置。
- `souldrive-ui/`：React + Tauri 桌面界面，用于展示知识文档、运行态、问答、证据链和导图。
- `tests/`：后端核心逻辑的 `unittest` 测试。
- `config/enterprise-policy.json`：本地运行策略，保留 API 限流、CORS、lite/license 等基础约束。

## 本地开发

后端建议使用名为 `souldrive` 的 Conda Python 环境：

```powershell
conda activate souldrive
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

完整检查：

```powershell
.\scripts\check.ps1
```

安全与依赖审计：

```powershell
.\scripts\audit.ps1
```

前端构建：

```powershell
cd souldrive-ui
npm install
npm run build
```

Tauri/Rust 检查：

```powershell
cd souldrive-ui\src-tauri
cargo check
```

桌面端会通过 Tauri 启动本地 sidecar，并把随机 API 端口和 API token 传给前端；前端不应写死默认端口作为桌面端主路径。
非公开本地 API 默认要求 `X-SoulDrive-Token`。仅本地调试时可设置 `SOULDRIVE_ALLOW_UNAUTHENTICATED_API=1` 临时放开。
`test.http` 中的 `{{souldrive_token}}` 需要替换为当前 sidecar 运行时 token，或在 IDE HTTP Client 环境变量里配置。

## 参赛便携形态

桌面端启动 sidecar 时会设置 `SOULDRIVE_WATCH_REMOVABLE=1`，后端会等待并接管已插入的移动存储工作区；工作区应位于可移动盘根目录下的 `SoulDrive/`。不要把固定盘符写入业务代码，盘符变化时由可移动盘探测逻辑处理。

推荐的 U 盘目录：

```text
<USB_DRIVE>:\
  SoulDrive.exe
  models\
  sidecars\
    souldrive-sidecar\
      souldrive-sidecar.exe
  SoulDrive\
    data\papers\
    index\
    audit\
    models\
    config\workspace.json
```

打包 Python sidecar：

```powershell
.\scripts\package-sidecar.ps1
```

该脚本会把 Docling、RapidOCR、ONNX Runtime、SentenceTransformers、llama.cpp Python 绑定等离线解析和推理依赖收集进 `souldrive-ui\src-tauri\sidecars\`。完成后再构建 Tauri 桌面端。

## 工作区约定

SoulDrive 会在本地或授权存储中维护 `SoulDrive/` 工作区：

- `data/papers/`：当前默认存放 PDF 知识文档，演示阶段仍以论文为主。
- `index/`：Chroma、父子切片索引、关键词索引。
- `graph/`：本地图数据库。
- `audit/`：hash-chain 审计日志。
- `models/`：本地 GGUF 聊天模型和 BGE 嵌入模型。

当前开发重点是先把这些内部链路做清楚：文档进入、索引更新、检索证据、回答生成、运行态锁定和审计记录。

## 运行产物与编码

- `souldrive_db/`、`models/`、`runtime/`、`node_modules/`、`souldrive-ui/dist/` 和 `souldrive-ui/src-tauri/target/` 都是本地运行或构建产物，不应提交。
- 中文源码和文档使用 UTF-8。PowerShell 读取中文文件时建议显式加 `-Encoding utf8`。
