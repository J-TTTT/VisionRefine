# 第一阶段：通用格式中心

本期是普通矩形检测框的跨工具交换，不是“所有任务、所有变体都已支持”。
任务类型、文件格式、标注审核版本仍是三个独立维度。统一内部坐标是原图像素 `xyxy`。

## 能力矩阵

| 适配器 ID | 双向交换范围 | 属性 / 划分与边界 |
|---|---|---|
| `images` | 仅导入图片，支持建立各任务清单 | 不把无标注图片视为已确认负样本 |
| `coco_detection` | COCO images/categories/annotations、bbox | `iscrowd`；按划分另附 JSON；不表示分割/关键点 |
| `yolo_detection` | Ultralytics YAML + 五列 TXT | 归一化 cx/cy/w/h；未指定划分转 train；不表示分割/姿态/OBB |
| `voc_detection` | VOC XML + ImageSets/Main | 整数框、difficult/truncated/occluded/pose；图片转 JPEG 可能有损 |
| `cvat_detection` | CVAT for images XML 1.0/1.1，导出 1.1 | 自定义属性转文本，occluded/z_order；不含视频 track、旋转框、mask |
| `label_studio_detection` | JSON task 数组，单版本 RectangleLabels | 百分比坐标；拒绝旋转、多版本歧义和远程图片下载；附标签配置 |
| `labelme_detection` | wkentaro Labelme 5.x JSON rectangle | 两点矩形，flags/group_id/description；不含多边形；不是 MIT LabelMe XML |
| `visionrefine` | 原生检测数据集快照 1.0 | 保留内部标注/状态/划分/置信度；清除本机路径与敏感元数据；不是项目备份 |

CVAT、Label Studio、Labelme 没有本期使用的标准划分字段，因此划分保存在
`visionrefine.json`。不能写入目标格式的属性和置信度也保存在该附加清单，不会假装
第三方工具已读取它们。能力注册表的 `tested` 表示当前声明范围通过仓库测试，
不等于每个历史/未来目标工具版本都兼容，更不等于生产稳定性承诺。

## 如何使用

1. 创建或追加项目时选择输入格式。新格式同样支持类别映射、重复图冲突与导入报告。
2. 保存人工标注。在导出区选择主格式，展开“同时导出更多格式”勾选其他格式。
3. 选择标注版本、一个或多个 train/val/test 划分、是否包含图片；可保存命名预设。
4. 点击“检查导出兼容性”，核对图像/类别/框数量与各格式的保留、转换、丢失项。
5. 逐项勾选允许的转换后提交。阻断项不能通过勾选绕过；需要修复或取消该格式后重新预览。
6. 任务区显示进度，可继续标注、刷新、取消未完成任务，或重试同一快照的全部格式。
   成功格式可单独下载；一批存在失败时保留成功包，并提供批量 ZIP。

预设只保存选项，不保存转换同意。每次实际导出都重新检查。后台任务使用预览时的一份
固定有效标注快照，后续人工保存不会混入该批次；重试仍用原快照，需要新结果时重新预览。
图片不预先复制冻结，但每次打包核对 SHA-256；预览后文件改变时该格式失败，绝不把新图片
与旧框混装。旧 manifest 尚无文件指纹时，第一次导出预览固定当前内容。
带非默认 EXIF 旋转/翻转的图片会阻断第三方格式导出，避免不同工具自动转向后框错位；
需先统一图片与标注方向，原生快照仍可保留原始内容。本期不自动变换 EXIF 坐标。

## 新格式输入与第三方打开方式

- CVAT：图片根目录对应 XML 中 `image name`，标注文件默认 `annotations.xml`。
  导出包的图片在 `images/`；在 CVAT 建立对应标签和图片任务，再选择 CVAT 1.1 导入 XML。
- Label Studio：根目录用于解析 `data.image` 相对路径或 `/data/local-files/?d=...`。
  人工确认版本导出到 `annotations`，AI/粗标注导出到 `predictions`，不冒充人工结果。
  应用包内 `label_config.xml`，启用本地文件服务并将 document root 指向解压根目录，
  再导入 `annotations.json`；详见包内 `IMPORT.md`。云端 URL 不会自动下载。
  多份有效 annotation/prediction 的任务必须先明确选一个版本，不能静默择一。
- Labelme：可指定单 JSON 或递归扫描标注目录，`imagePath` 优先相对于 JSON 解析，
  但解析结果必须仍在选定图片根目录内。相对 `../` 仅允许用于根目录内的兄弟目录。
  当前需要本地图片，不仅凭内嵌 `imageData` 导入。导出图片与 JSON 同处 `images/`，
  用 Labelme 打开此目录。使用哈希文件名避免不同扩展名的同名图片覆盖，映射在报告中。
- 原生：解压后选择根目录和 `dataset.json`；或选择原生 ZIP，根目录填写 ZIP 所在目录
  （ZIP 模式实际会重新绑定到独立缓存）。不把普通 `visionrefine.json` 审计侧车当作包。
  仅标注包必须先按映射补齐图片才能重新导入。

空标注的数据集可以使用 `classes.json`（类别字符串数组）提供未使用类别；Labelme /
Label Studio 无类别文件时从对象标签发现类别，完全空时再使用用户填写的备用类别。

## 原生包信任策略（已确认）

默认将非未标注状态降为 `imported_coarse`，并在 provenance 中记录原审核状态。
只有用户勾选“信任原生包并恢复其审核状态”，才恢复包中状态，允许原来人工确认的图像
进入 `reviewed` 导出；该决定写入导入历史。导入包本身没有真实性签名，勾选代表用户信任
来源，不是系统替用户验证过标注质量。

无论是否信任，追加重复图都不能覆盖已有人工/AI 文档；已经作为可信审核状态导入的
manifest 记录也受保护。原生包只包含本次选定标注快照，不恢复全项目修订历史、模型连接、
API Key、任务队列或用户账户。不会把工作目录或绝对源路径打包。

## 本地任务与安全边界

- 单服务进程、同一 workspace；最多两个正在执行的导出。不是多进程/分布式任务系统。
- `export-plans/` 保存私有固定快照；`export-jobs/` 保存原子写入的状态和批量 ZIP。
  导入/导出永远只读取原始数据，图片副本只写到 workspace 或指定测试目录。
- 服务重启后第一次访问任务 API 会将未完成任务标为 `interrupted`，需显式重试。
  已完成下载仍可用；不承诺逐字节断点续传。取消是检查点协作式，不保证立即结束大型解析。
- 只有最终 ZIP 原子发布后才提供下载；失败/取消残留目录或 `.part` 不可下载。
  当前没有自动过期清理，需管理 workspace 空间；预览也会读取图片内容，暂非后台任务。
- 原生 ZIP 拒绝绝对/越界路径、反斜杠、符号链接/特殊文件、大小写冲突、加密成员及
  可疑压缩比；最多 100,000 成员、展开总量 20 GiB。只解压到全新独立缓存。
- 敏感键/路径脱敏会在预检查列为需确认的信息损失。它是防误泄露措施，不是通用 DLP；
  自定义自由文本如果包含无法识别的隐私，发布前仍需人工核对。
- 旧同步 `/dataset/export` API 为兼容保留，仍是原来的 warning-only 行为。
  新接入方应使用以下预览/确认/任务 API；不要把旧接口当作强制转换审批边界。

## API

以下路径均位于 `/api/projects/{project_id}`：

| 方法与路径 | 输入 / 返回 |
|---|---|
| POST `/dataset/export/preview` | `{formats, policy, splits, include_images}` → 固定 preview_id、数量、issues |
| POST `/dataset/export/jobs` | `{preview_id, acknowledgements: {format_id: [issue_code]}}` → 202 任务 |
| GET `/dataset/export/jobs` | 已持久化任务，最新在前 |
| GET `/dataset/export/jobs/{job_id}` | 状态、逐格式进度/错误/下载链接 |
| POST `/dataset/export/jobs/{job_id}/cancel` | 请求取消剩余工作 |
| POST `/dataset/export/jobs/{job_id}/retry` | 对相同快照创建新任务，重跑所有格式 |
| GET `/dataset/export/jobs/{job_id}/download` | 已完成的批量 ZIP，未完成返回 409 |
| GET / POST `/dataset/export/presets` | 读取或保存 `{name, options}`；同名替换，最多 50 个 |

`issues` 包括 `info / conversion / loss / blocking`。conversion/loss 每项需明确确认。
相同 preview_id 重复提交返回同一任务，重试接口除外。`formats` 不受项目输入格式限制。
原生导入来源新增 `trust_reviewed: false`，仅对 `visionrefine` 生效。

## 开发新适配器

1. 复制 [适配器模板](adapter-template.py)，实现 `read` / `write`，必要时添加 `preflight`。
2. 在注册表声明版本、任务、几何、可表示属性、置信度/划分能力、输入字段、限制和稳定性。
   界面自动读取 `/api/dataset-formats`，无需新增按格式 ID 分支的导出按钮。
3. 只处理格式转换；版本筛选、快照、拷图、ZIP、进度、状态持久化归通用服务层。
4. 用 `dataset_io.testing.detection_fixture` 生成独立测试源，用
   `assert_detection_equivalent` 比较类别、逐图框、显式空标注，可明确设置 VOC 取整容差。
5. 覆盖 Unicode/嵌套路径、空图、未使用类别、小数框、坏行/越界、未知几何、丢失属性与负例。
   每个静默丢弃风险都必须变成导入报告、导出确认或阻断项。
6. 增加跨格式和浏览器回归，再用独立解析器/目标工具验证。不能只做自己读自己写。

生成七格式合成样例（目标目录必须不存在）：

```bash
.venv/bin/python -m scripts.generate_format_samples /tmp/visionrefine-samples
VISIONREFINE_BROWSER_TESTS=1 .venv/bin/python -m pytest -q
```

独立解析器检查可以在单独虚拟环境安装 `datumaro==1.13.11`、`labelme==5.5.0`、
`label-studio-converter==0.0.59`，运行：

```bash
python scripts/verify_format_samples.py /tmp/visionrefine-samples
```

使用 Datumaro 的 CVAT 读取器、Labelme 官方 LabelFile、Label Studio 官方 converter、
COCO pycocotools 对合成样例验证类别/坐标/图像数。独立解析器冒烟不等于已运行
CVAT / Label Studio 完整服务器 UI，也不能替代目标环境实际验收。

`tests/test_format_matrix.py` 另覆盖 7×7 全部 49 种输入/输出组合，逐图比较类别、坐标、
空标注和划分；VOC 仅允许已声明的向外取整误差。浏览器测试覆盖类别精修、追加保护、
逐项转换确认、多格式下载、预设刷新恢复以及原生包默认不信任。

格式依据：[CVAT XML](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/)、
[Label Studio 预标注](https://labelstud.io/guide/predictions)、
[Labelme](https://github.com/wkentaro/labelme)、
[Label Studio converter](https://github.com/HumanSignal/label-studio-converter)。
