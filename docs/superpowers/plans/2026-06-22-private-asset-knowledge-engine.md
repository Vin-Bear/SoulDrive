# Private Asset Knowledge Engine Repositioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reposition SoulDrive from a paper-specific workbench into a private-asset knowledge engine by generalizing API semantics, frontend naming, and user-facing copy while preserving paper-first demos and existing runtime protocols.

**Architecture:** Keep the current Python sidecar, workspace layout, and runtime protocol intact. Add `documents` API semantics as the primary interface, leave `papers` routes as compatibility shims, and update the React desktop UI plus product docs to describe a broader private-asset knowledge engine that still uses papers as the default demo entry point.

**Tech Stack:** Python `unittest`, FastAPI, React 19 + TypeScript, Tauri, static HTML docs

---

## File Structure Map

**Modify:**

- `D:\PycharmProjects\LangChainProjects\SoulDrive\core\mcp_server.py`
  Purpose: expose `/documents/*` routes, keep `/papers/*` compatibility, rename response payload fields to document semantics.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive-ui\src\App.tsx`
  Purpose: switch frontend state/types/requests from paper semantics to document semantics and rewrite user-facing copy.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\README.md`
  Purpose: update product positioning and API examples.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive_project_explainer.html`
  Purpose: update architecture narrative from paper workbench to private-asset knowledge engine.
- `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_mcp_server_papers.py`
  Purpose: validate the new `/documents/*` routes and keep explicit coverage for old `/papers/*` compatibility.

**Keep unchanged on purpose:**

- `D:\PycharmProjects\LangChainProjects\SoulDrive\core\paper_importer.py`
- `D:\PycharmProjects\LangChainProjects\SoulDrive\core\workspace.py`
- `D:\PycharmProjects\LangChainProjects\SoulDrive\config\enterprise-policy.json`
- `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive-ui\src\App.css`

These files keep the current storage conventions, protocol prefixes, and CSS class names to avoid needless churn.

### Task 1: Add Document-Semantic API Surface With Compatibility

**Files:**
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_mcp_server_papers.py`
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\core\mcp_server.py`

- [ ] **Step 1: Write the failing API tests for `/documents/*`**

Add document-semantic assertions without removing the existing compatibility coverage.

```python
    def test_documents_list_reads_only_workspace_documents_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            managed = Path(workspace.papers_path) / "managed.pdf"
            managed.parent.mkdir(parents=True, exist_ok=True)
            managed.write_bytes(b"%PDF-1.4\n%managed\n")

            with sqlite3.connect(workspace.parent_doc_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS parent_documents (
                        parent_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "INSERT OR REPLACE INTO parent_documents (parent_id, content, metadata_json) VALUES (?, ?, ?)",
                    ("managed_parent", "content", json.dumps({"source_filename": "managed.pdf"})),
                )
                conn.commit()

            with TestClient(mcp_server.app) as client:
                response = client.get(
                    "/documents/list",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["document_count"], 1)
        self.assertEqual(payload["documents"][0]["name"], "managed.pdf")
        self.assertTrue(payload["documents"][0]["indexed"])

    def test_documents_import_copies_pdf_into_active_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            source_dir = Path(temp_dir) / "incoming"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_pdf = source_dir / "paper.pdf"
            source_pdf.write_bytes(b"%PDF-1.4\n%import\n")

            with TestClient(mcp_server.app) as client:
                response = client.post(
                    "/documents/import",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"source_paths": [str(source_pdf)]},
                )
                imported_exists = (Path(workspace.papers_path) / "paper.pdf").exists()

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["imported_count"], 1)
        self.assertTrue(imported_exists)
        self.assertEqual(payload["items"][0]["status"], "imported")
```

- [ ] **Step 2: Run the API tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_mcp_server_papers.McpServerPapersTests.test_documents_list_reads_only_workspace_documents_directory tests.test_mcp_server_papers.McpServerPapersTests.test_documents_import_copies_pdf_into_active_workspace -v
```

Expected:

- `404` or missing-key failures because `/documents/*` and `document_count` / `documents` do not exist yet.

- [ ] **Step 3: Implement shared document response helpers and new routes**

Refactor the existing paper handlers so one implementation feeds both the new routes and the compatibility routes.

```python
def _document_library_payload(workspace: SoulDriveWorkspace) -> dict:
    documents_dir = Path(workspace.papers_path)
    indexed_sources = _indexed_source_filenames(workspace)
    documents = []

    for path in sorted(documents_dir.rglob("*.pdf"), key=lambda item: item.name.lower()):
        stat = path.stat()
        documents.append({
            "name": path.name,
            "relative_path": path.relative_to(documents_dir).as_posix(),
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "indexed": path.name in indexed_sources,
        })

    return {
        "ready": True,
        "document_count": len(documents),
        "indexed_count": sum(1 for document in documents if document["indexed"]),
        "workspace": "SoulDrive workspace mounted",
        "documents": documents,
    }


def _import_documents(request: PaperImportRequest):
    state = get_runtime_state()
    if state.get("locked"):
        return JSONResponse(
            {"error": "SoulDrive workspace is locked", "status": "locked"},
            status_code=423,
        )

    workspace = current_workspace()
    items = [import_paper_into_workspace(workspace, source_path) for source_path in request.source_paths]
    return {
        "ready": True,
        "imported_count": sum(1 for item in items if item["status"] == "imported"),
        "items": items,
        "workspace": "SoulDrive workspace mounted",
    }


@app.get("/documents/list")
async def documents_list():
    workspace = current_workspace()
    return _document_library_payload(workspace)


@app.post("/documents/import")
async def documents_import(request: PaperImportRequest):
    return _import_documents(request)


@app.get("/papers/list")
async def papers_list():
    payload = _document_library_payload(current_workspace())
    return {
        **payload,
        "paper_count": payload["document_count"],
        "papers": payload["documents"],
    }


@app.post("/papers/import")
async def papers_import(request: PaperImportRequest):
    return _import_documents(request)
```

- [ ] **Step 4: Add one compatibility assertion for the old `/papers/list` shape**

Keep one old-route assertion in the same test file so the compatibility contract stays explicit.

```python
    def test_papers_list_keeps_legacy_response_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            managed = Path(workspace.papers_path) / "legacy.pdf"
            managed.parent.mkdir(parents=True, exist_ok=True)
            managed.write_bytes(b"%PDF-1.4\n%legacy\n")

            with TestClient(mcp_server.app) as client:
                response = client.get(
                    "/papers/list",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("paper_count", payload)
        self.assertIn("papers", payload)
```

- [ ] **Step 5: Run the route tests to verify they pass**

Run:

```powershell
python -m unittest tests.test_mcp_server_papers -v
```

Expected:

- `OK`
- New `/documents/*` tests pass.
- Legacy `/papers/*` compatibility assertions pass.

- [ ] **Step 6: Commit the backend/API change**

Run:

```powershell
git add core/mcp_server.py tests/test_mcp_server_papers.py
git commit -m "feat: add document-semantic knowledge API routes"
```

### Task 2: Switch The Desktop UI To Document Semantics

**Files:**
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive-ui\src\App.tsx`

- [ ] **Step 1: Write the failing frontend type/build change**

Rename the TypeScript interfaces and state variables in one pass so the compiler shows every remaining paper-specific reference.

```tsx
interface DocumentItem {
  name: string;
  relative_path: string;
  size_bytes: number;
  modified_at: number;
  indexed: boolean;
}

interface DocumentLibrary {
  ready: boolean;
  document_count: number;
  indexed_count: number;
  workspace: string;
  documents: DocumentItem[];
}

interface DocumentImportResponse {
  imported_count: number;
  items: Array<{
    name: string;
    status: "imported" | "already_present" | "rejected";
    error_code?: string;
  }>;
}
```

Update state declarations to the new names:

```tsx
  const [documentLibrary, setDocumentLibrary] = useState<DocumentLibrary | null>(null);
  const [documentPage, setDocumentPage] = useState(1);
  const [documentImportStatus, setDocumentImportStatus] = useState<"idle" | "selecting" | "importing" | "error">("idle");
  const [documentImportMessage, setDocumentImportMessage] = useState("");
```

- [ ] **Step 2: Run the frontend build to verify the rename breaks existing references**

Run:

```powershell
cd souldrive-ui
npm run build
```

Expected:

- TypeScript errors for `PaperItem`, `PaperLibrary`, `paperLibrary`, `paperPage`, `paperImportStatus`, or old payload field names.

- [ ] **Step 3: Update requests and data access to `/documents/*`**

Switch all list/import fetches and derived state to `documents`.

```tsx
    const refreshDocumentLibrary = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/documents/list`, {
          headers: apiToken ? { "X-SoulDrive-Token": apiToken } : undefined,
        });
        if (!response.ok) throw new Error("documents unavailable");
        const data = await response.json();
        if (isMounted) setDocumentLibrary(data);
      } catch {
        if (isMounted) setDocumentLibrary(null);
      }
    };
```

```tsx
      const response = await fetch(`${apiBaseUrl}/documents/import`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiToken ? { "X-SoulDrive-Token": apiToken } : {}),
        },
        body: JSON.stringify({ source_paths: sourcePaths }),
      });
      if (!response.ok) throw new Error("document import failed");

      const payload = await response.json() as DocumentImportResponse;
      setDocumentImportMessage(`导入 ${payload.imported_count}/${payload.items.length} 份`);
```

Derived state should read:

```tsx
  const documents = documentLibrary?.documents ?? [];
  const totalDocumentPages = Math.max(1, Math.ceil(documents.length / PAPERS_PER_PAGE));
  const activeDocumentPage = Math.min(documentPage, totalDocumentPages);
  const visibleDocuments = useMemo(() => {
    const start = (activeDocumentPage - 1) * PAPERS_PER_PAGE;
    return documents.slice(start, start + PAPERS_PER_PAGE);
  }, [activeDocumentPage, documents]);
```

- [ ] **Step 4: Rewrite visible copy while keeping papers as the first demo entry**

Use broader knowledge-engine language, but keep examples and templates paper-friendly enough for demos.

```tsx
const promptTemplates = [
  "请归纳这批论文或技术资料的核心主题、关键结论和差异，并给出引用依据。",
  "请对比近三份技术文档、研究论文或项目方案的目标、实现路径、约束和风险。",
  "请生成一个围绕当前资料库的技术路线图，并标注关键来源。",
  "请从企业落地角度评估这些资料对应方案的可复用性、成本和实施风险。",
];
```

```tsx
            <h1>灵枢 SoulDrive</h1>
            <span>面向私有资产的端侧知识引擎</span>
```

```tsx
                知识文档
```

```tsx
                <span>份文档</span>
```

```tsx
              <h2>私有知识问答与分析</h2>
```

```tsx
                Local Knowledge Engine
```

```tsx
              placeholder={locked ? "授权移动存储未就绪，知识引擎已锁定" : isGenerating ? "端侧模型生成中..." : "输入文档归纳、对比、方案分析或问答任务"}
```

Empty-state and evidence copy should become:

```tsx
                  <span>暂无文档</span>
```

```tsx
        <p>完成一次知识问答后，这里会显示检索片段、来源文件和重排分。</p>
```

- [ ] **Step 5: Run the frontend build to verify the UI compiles**

Run:

```powershell
cd souldrive-ui
npm run build
```

Expected:

- `vite build` completes successfully.

- [ ] **Step 6: Commit the frontend rename**

Run:

```powershell
git add souldrive-ui/src/App.tsx
git commit -m "feat: reposition desktop UI as knowledge engine"
```

### Task 3: Update Product Docs And Architecture Narrative

**Files:**
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\README.md`
- Modify: `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive_project_explainer.html`

- [ ] **Step 1: Rewrite the README opening and module descriptions**

Update the opening copy so it describes a private-asset knowledge engine, while explicitly saying papers remain the default demo material.

```md
# SoulDrive

SoulDrive 是一个面向私有资产的端侧知识引擎，核心目标是把本地知识文档的解析、索引、检索增强生成、证据链和知识图谱打磨稳定。当前演示与测试仍以论文作为默认切入口，但系统能力不局限于论文场景。

## 核心模块

- `core/`：本地 sidecar、工作区管理、文档解析、索引、检索、RAG、审计日志、模型运行配置。
- `souldrive-ui/`：React + Tauri 桌面界面，用于展示知识文档、运行态、问答、证据链和导图。
```

Also update the workspace section:

```md
- `data/papers/`：当前默认存放 PDF 知识文档，演示阶段仍以论文为主。
```

- [ ] **Step 2: Update the explainer HTML title, summary, and section wording**

Change the static explainer to match the new positioning while preserving the accurate architecture.

```html
<title>SoulDrive 项目速读与架构讲解</title>
...
<h1>SoulDrive 项目速读与架构讲解</h1>
<p class="subtitle">
  这份说明基于当前源码结构整理，目标是帮助你快速建立项目全局图景：SoulDrive 是一个面向私有资产的端侧知识引擎，
  当前测试与演示仍以论文作为第一切入口，但底层能力已覆盖更广泛的本地知识文档场景。
</p>
```

Wherever the explainer says “论文工作台” or “论文库”, replace it with “知识引擎”, “知识文档”, or “私有知识文档” as appropriate. Keep the architecture accurate by noting that:

```html
<strong>论文仍然是默认演示入口</strong>
```

- [ ] **Step 3: Run a doc sanity check**

Run:

```powershell
rg -n "端侧论文知识工作台|论文库|Local Paper RAG|论文归纳与智能问答" README.md souldrive_project_explainer.html souldrive-ui/src/App.tsx
```

Expected:

- No remaining stale product-positioning strings.
- Any remaining `论文` text should be intentional demo-context wording inside templates or explanatory copy.

- [ ] **Step 4: Commit the doc refresh**

Run:

```powershell
git add README.md souldrive_project_explainer.html
git commit -m "docs: reposition SoulDrive as private asset knowledge engine"
```

### Task 4: Run Full Regression And Final Cleanup

**Files:**
- Modify if needed: `D:\PycharmProjects\LangChainProjects\SoulDrive\tests\test_mcp_server_papers.py`
- Modify if needed: `D:\PycharmProjects\LangChainProjects\SoulDrive\souldrive-ui\src\App.tsx`
- Verify: `D:\PycharmProjects\LangChainProjects\SoulDrive\README.md`
- Verify: `D:\PycharmProjects\LangChainProjects\SoulDrive\core\mcp_server.py`

- [ ] **Step 1: Run the full Python test suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected:

- `OK`
- No regressions in route, runtime, retrieval, or diagnostics tests.

- [ ] **Step 2: Run the frontend production build from a clean working directory**

Run:

```powershell
cd souldrive-ui
npm run build
```

Expected:

- `✓ built in ...`

- [ ] **Step 3: Check for stale public paper-specific API strings**

Run:

```powershell
rg -n "\"/papers/list|\"/papers/import|paper_count|PaperItem|PaperLibrary|PaperImportResponse|Local Paper RAG|端侧论文知识工作台" core souldrive-ui/src README.md souldrive_project_explainer.html tests
```

Expected:

- Remaining `papers` strings only appear in compatibility routes, storage-path comments, or intentional backward-compatibility tests.

- [ ] **Step 4: Review git diff for scope control**

Run:

```powershell
git diff -- core/mcp_server.py souldrive-ui/src/App.tsx README.md souldrive_project_explainer.html tests/test_mcp_server_papers.py
```

Expected:

- Only the planned product-positioning, route-semantics, and compatibility edits appear.
- No protocol-prefix, workspace-root, or unrelated refactors appear.

- [ ] **Step 5: Commit the regression pass cleanup**

Run:

```powershell
git add core/mcp_server.py souldrive-ui/src/App.tsx README.md souldrive_project_explainer.html tests/test_mcp_server_papers.py
git commit -m "test: verify knowledge engine repositioning changes"
```

## Spec Coverage Check

- Product定位升级：covered by Task 2 and Task 3.
- `documents` API 语义：covered by Task 1.
- 旧 `papers` 路径兼容：covered by Task 1 and Task 4.
- 前端状态/类型/文案收口：covered by Task 2.
- README 与架构讲解同步：covered by Task 3.
- 论文仍为默认演示入口：covered by Task 2 prompt copy and Task 3 README/explainer wording.

## Placeholder Scan

- No `TBD`, `TODO`, or deferred implementation markers remain.
- Each test/build command is explicit.
- Each code-writing step includes the concrete snippet to add or replace.

## Type Consistency Check

- `DocumentItem`, `DocumentLibrary`, and `DocumentImportResponse` are used consistently across the frontend tasks.
- `document_count` / `documents` are the new primary response fields.
- `/papers/*` remains compatibility-only and is not reused as the primary UI path.
