# Dataset I/O

VisionRefine separates task type, dataset format and revision source. Importers produce
a common dataset manifest; the editor and AI workflow use original-image pixel boxes;
exporters consume a selected snapshot in that same model.

The detection adapters are **COCO**, **YOLO**, **Pascal VOC**, **CVAT for images**,
**Label Studio RectangleLabels**, **Labelme rectangles**, and **VisionRefine native snapshots**.
All can import and export independently of the original format.
Images-only projects can export their human annotations through any adapter.
Multi-source merge, import preview, explicit category mapping, and append imports
share the same format-independent transaction layer. Non-detection annotation adapters
remain future extensions. See [the format center guide](format-center.md) for the current
capability matrix, compatibility consent, export jobs, native trust and extension templates.

## 使用方法

1. 创建项目，选择目标检测和数据集导入格式。
2. 仅图片：指定图像根目录，选择或添加自定义标签。
3. COCO Detection：指定图像根目录与 JSON 文件；`file_name` 相对于图像根目录解析。
   类别来自 `categories`，支持非连续类别 ID 和中文名称。选择该文件的默认划分。
4. YOLO Detection：指定包含 `images/`、`labels/` 的根目录和 `data.yaml`。
   YAML 的 `train`、`val`、`test` 支持目录、目录列表和图片路径 TXT 列表；类别来自 `names`。
5. Pascal VOC Detection：根目录包含 `JPEGImages/`、`Annotations/`、`ImageSets/Main/`；
   标注目录可留空。类别从 XML 和可选 `classes.txt` 读取。
6. 点击“预览并导入数据”，或“配置多来源 / 多划分导入”添加更多来源。
   每行指定一个独立来源标识、目录、格式、标注输入和默认划分。
7. 预览后核对数量、数据划分与来源检查报告；在各来源下修改“原类别 → 项目类别”映射。
   同名类别自动合并，`pedestrian → person` 等异名映射必须明确指定。
   映射或重复图片策略修改后，需要重新预览，确认按钮才会启用。
8. 确认创建并导入后，打开工作台检查和修改。保存操作生成独立人工版本。
9. 在“数据集导入 / 导出”选择一个或多个目标格式、划分、版本和图片选项，点击
   “检查导出兼容性”，逐项确认转换后后台生成 ZIP。预设与任务历史会保存。
10. 后续批次点击“追加数据”，复用同一预览流程。“导入历史”可以查看各批次来源、
    类别映射、冲突处理、数量及父版本，不会自动回滚数据。

导入标注后类别集合锁定，以保证已有类别引用一致。仅图片项目可以在初次保存或 AI
推理前调整类别。已有导入标注或人工/AI 版本的项目不支持直接替换数据集，但支持安全追加。
追加时可将传入类别映射到已有类别，或在末尾添加新类别；不能借此重命名已有类别，
已有类别 ID 保持不变。新来源的图像使用 `来源标识/原相对路径` 作为稳定工作台键。
旧项目首次追加时补充来源信息，但保持旧图像键不变，原有人工标注继续有效。

### 合并、重复与提交规则

- 同一项目可以混合多份 COCO JSON、YOLO YAML、VOC 目录及仅图片来源，各来源可位于
  不同本地目录。图片不会为了导入而复制；相同文件名、不同内容的图像可同时存在。
- 导入不扫描整个来源目录计算文件指纹，也不按图片内容去重。仅相同的来源标识与相对路径
  视为同一张图；不同来源即使指向相同文件，也按两张图处理。用户负责图片、标注和划分的对应关系。
- 同一路径冲突默认“保留原数据”，也可逐张选择“更新导入粗标注”或“阻止提交”。只更新
  manifest 中的粗标注，不修改 `annotations/`、`suggestions/` 或人工历史；显式人工空标注
  仍然优先。仅图片来源不能清空已有粗标注。
- 同一路径冲突始终保留原划分；不同来源之间的重复图片与划分由用户自行检查。
  冲突表展示传入与原有划分、框数量及采用的策略。无差异的重复也会报告。
- 来源标识用于命名空间，同一标识不能改指另一目录/标注输入。追加界面会自动生成新标识；
  有意复用来源时，可以输入与原来相同的标识。
- 预览只保存临时计划，不创建或改动项目。预览解析格式并确认引用图片可读取；
  提交再次确认这些图片仍然存在，并检查项目类别集合与数据集版本。来源内容变化
  不会被自动检测，预览后修改标注或替换图片时请重新预览。
  期间保存人工标注是允许的，提交会保留它。
- 提交发布新的不可变 manifest，通过原子替换项目指针使其生效；同一预览重复提交不会
  重复追加。项目级锁防止本地服务内并发追加/人工保存相互覆盖。
- 部分坏图/坏框可以跳过，错误必须在预览报告中确认；没有任何可读图像或类别目录非法
  则不允许导入。发现重复并选择“阻止提交”时也不会改动项目。
- 预览和提交是同步操作。格式解析仍会读取标注和图片尺寸；仅图片模式以及部分格式
  也会按该格式要求枚举图片，大数据集仍可能耗时较长。当前支持单进程本地服务，不支持多个服务进程
  共同写同一个 workspace。不要将 workspace 放在数据集根目录内。
- 图片继续引用本地源文件，并非内容冻结副本；导入后不要移动、覆盖源文件。
  导出含图片的 ZIP 可用于可搬移的独立数据包。

### Export policies

| Policy | Included effective versions |
|---|---|
| `reviewed` (default) | Human-reviewed images, including explicitly reviewed empty images |
| `reviewed_or_ai` | Human-reviewed images, then AI suggestions for remaining images |
| `all_annotated` | Human-reviewed images, then AI suggestions, then imported coarse annotations |

An unreviewed images-only image is excluded, not silently treated as a verified negative.
An imported empty COCO image or missing YOLO label file is a **coarse** negative, never
a human-reviewed negative. YOLO missing label files are reported because a missing file
can also indicate a dataset layout mistake. Import errors and dropped fields remain
visible in the report and export sidecar.

Saving an empty human version takes precedence over nonempty AI/imported versions.
Model calls and AI suggestions retain the existing behavior; Dataset I/O does not
implement selective VLM review or final AI auditing.

## Internal contract

`visionrefine/core/dataset_io/models.py` defines the versioned Pydantic schema:

- Dataset: `schema_version`, `task`, `categories`, `images`, `sources`, `provenance`, `report`.
- Category: canonical integer `id`, `name`, original `source_id` and provenance.
- Image: unique `id`, relative `path`, actual `width`/`height`, `split`, `status`, objects.
  Schema 2.0 adds `source_id`, `source_path` and an optional file-byte `sha256`; `path` is a logical
  stable key, not necessarily a physical relative path. Schema 1.0 remains readable.
- Source: unique `id`, absolute local `root`, `format`, optional `annotation_path`, provenance.
- Annotation v1: `kind=bbox`, `category_id`, `label`, pixel `bbox=[x1,y1,x2,y2]`,
  optional confidence, source, attributes and provenance.
- Splits: `train`, `val`, `test`, `unspecified`. No random split is inferred.
- Revision source: `imported_coarse`, `ai_suggestion`, `human_reviewed`.
  `final_reviewed` is reserved in the schema; this release has no final-audit workflow.

`task` is independent of the adapter ID. Adding another task requires a corresponding
annotation schema/editor and an adapter declaring support for that task; adding a
format for existing detection data requires only an adapter and registration.

The original import manifest is stored under
`workspace/projects/<project>/datasets/import-<id>.json`. It is not rewritten by editing.
Legacy single-source image-only reanalysis creates a new manifest of the source directory.
Preview-based datasets retain their explicit manifest on reanalysis; add files through
the append workflow. The complete image list is retained, including datasets larger
than 200 images. Manifests link `parent_revision_id` and contain import operation records.
Temporary preview plans live in `workspace/import-previews/`; they are not dataset versions.

AI and human documents remain in the existing `suggestions/`, `annotations/` and
`revisions/` directories. Human saves record a parent revision and preserve attributes
and provenance. Source files and images are never modified by import/export.

## Format behavior and conversion boundaries

### COCO Detection

- Requires `images`, `categories`, `annotations` arrays; import only references listed
  images. Unlisted files in the image root are not added to the project.
- Converts `xywh` to pixel `xyxy`; finite positive boxes are clipped to image bounds
  with a warning. Invalid annotations and missing/unreadable images are skipped with errors.
- Category IDs must be unique. Duplicate category names are merged with a warning;
  original aliases remain in provenance. Invalid category catalogs fail the import.
- Actual local image sizes take precedence over inconsistent metadata, with a warning.
- Retains `iscrowd`; segmentation, keypoints and other unsupported fields are omitted
  with explicit report entries. This adapter does not convert masks into boxes.
- Export preserves source category IDs when possible; annotation/image IDs are regenerated.
  `area` is the bounding-box area. Confidence and workflow metadata live in the sidecar.
- Export includes a combined `annotations.json` and standard per-split files
  `annotations/instances_<split>.json`. COCO has no standard split field; use those
  separate files with the corresponding split option when re-importing.
- COCO image files are addressed relative to the export's `images/` directory.

### YOLO Detection

- Uses safe YAML loading and never executes a YAML `download` command.
- The explicitly chosen dataset root is authoritative. If YAML `path` refers to a
  different location, the import reports a relocation warning and uses the selected root.
- Absolute split/list entries are accepted only inside that root. List entries beginning
  `./` are resolved relative to the TXT file; other relative entries use the dataset root.
- The final `images/` path component maps to `labels/`; otherwise a sibling `.txt`
  is used. Multiple source images sharing one label path fail as ambiguous.
- `names` supports a list or a contiguous zero-based index/name mapping. Detection rows
  require exactly five fields: class index and normalized `cx cy width height`.
  Polygon, pose, non-finite, out-of-range and unknown-class rows are reported and skipped.
- An image cannot belong to multiple splits. A missing label file is reported and
  represents an imported coarse negative according to YOLO conventions.
- Export remaps classes to contiguous zero-based indices and includes `data.yaml`,
  per-split image lists, and `labels/<split>/.../*.txt`. It writes empty files for
  selected empty annotations. Filename stem collisions are disambiguated and mapped.
- `unspecified` images go to `train` with a warning. A validation set is never invented;
  training tools may require users to supply missing `train` or `val` splits.
- Attributes such as crowd flags cannot be represented by YOLO detection text; warnings
  explain the loss and `visionrefine.json` retains the metadata.

### Pascal VOC Detection

- Reads one XML per image in `Annotations/` (or the chosen annotation directory).
  `<filename>` is resolved inside `JPEGImages/`; external `<path>` is not followed.
  DTD/entity declarations are rejected. Traversal and escaping symlinks are rejected.
- `ImageSets/Main/train.txt`, `val.txt`, `test.txt` and optional `unspecified.txt` assign
  splits by XML filename stem. `trainval.txt` is an aggregate, not another split.
  Unlisted XML files use the configured default split. Missing XML/image pairs are reported;
  images with no XML are not silently imported as negatives. Use image-only import for them.
- Boxes use **one-based inclusive** VOC coordinates: `[xmin-1, ymin-1, xmax, ymax]` becomes
  internal zero-based pixel edges. One-pixel boxes remain valid. Zero-based custom VOC
  variants must be normalized before import; they are not auto-detected.
- Retains `difficult`, `truncated`, `occluded` and `pose` as object attributes.
  Segmentation/non-detection fields are reported, not converted. Empty XML annotations
  are coarse negatives; `classes.txt` or supplied labels can define an empty dataset's catalog.
- Export writes `Annotations/*.xml`, `ImageSets/Main/<split>.txt`, aggregate `trainval.txt`,
  and `classes.txt`. Stable hashed image IDs avoid basename collisions. Unspecified split
  is retained in `unspecified.txt`; an external training loader may require choosing a split.
- Fractional box edges are rounded outward to enclosing integer pixels, with a warning.
  Unsupported attributes, confidence and revision details remain in the sidecar.
- Image destinations are always `JPEGImages/*.jpg` for compatibility with standard VOC
  loaders. Included non-JPEG images are converted to RGB JPEG (quality 95), which can be
  lossy; source files remain unchanged. For annotations-only exports, **convert** non-JPEG
  originals to JPEG before populating the mapped paths; changing the extension is insufficient.

With **annotations only**, the ZIP contains no source images. `export-report.json`
maps every logical image key to its required output path. Portable sidecars do not contain
absolute source roots or model credentials. Populate those output locations
before standalone use. With **include images**, the archive contains copied images in
the required layout. Exports are stored under the project's `exports/` directory and
use distinct IDs, so repeated exports do not overwrite each other.

Every package includes `export-report.json` (category/path mappings, selected revisions,
skips, warnings) and `visionrefine.json` (selected internal annotations and original
import report). The sidecar is for auditing. Select the native adapter to produce an
importable `dataset.json` envelope, optionally with images. Only **native ZIPs** support
guarded extraction into an independent cache; other formats must be unpacked locally.
Export jobs run in the background; import preview/commit remains synchronous.
Remote downloads, training, full project backup and multi-process workers are not implemented.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/dataset-formats` | Task/format capabilities from the registry |
| POST | `/api/dataset-imports/preview` | Preview new/append multi-source import and receive a preview ID |
| POST | `/api/dataset-imports/{preview_id}/commit` | Revalidate and commit the exact preview, idempotently |
| POST | `/api/projects` | Create with `dataset_format`, `annotation_path`, optional `split` |
| POST | `/api/projects/{id}/analyze` | Refresh dimensions/routing and create a legacy/image-only manifest |
| GET | `/api/projects/{id}/dataset` | Original import manifest and full report |
| GET | `/api/projects/{id}/dataset/history` | Import/append version chain, newest first |
| POST | `/api/projects/{id}/dataset/import` | Attach annotations to an untouched images-only project |
| POST | `/api/projects/{id}/dataset/export` | Build package and return report/download URL |
| GET | `/api/projects/{id}/dataset/exports/{export_id}` | Download the completed ZIP |

Example preview body (omit `project_id` to create a new project):

```json
{
  "name": "People detection",
  "task": "detection",
  "duplicate_policy": "keep_existing",
  "sources": [
    {
      "id": "train2026", "root": "/data/train/images",
      "format": "coco_detection", "annotation_path": "/data/train/instances.json",
      "split": "train", "category_mapping": {"pedestrian": "person"}
    },
    {
      "id": "val2026", "root": "/data/validation",
      "format": "yolo_detection", "annotation_path": "/data/validation/data.yaml"
    }
  ]
}
```

For append, add `"project_id": "<existing-project-id>"`; the existing task is authoritative.
`duplicate_policy` is `keep_existing`, `update_coarse`, or `error`.
Optional `conflict_resolutions` maps incoming logical image keys to per-image policies,
e.g. `{"val2026/images/val/a.jpg": "update_coarse"}`. Image-only sources supply `labels`.
Review `sources[].report`, `sources[].mappings`, `conflicts`, `summary` and `commit_allowed`
in the response before POSTing the returned `preview_id` to the commit endpoint.
After a successful commit, POST `/api/projects/{id}/analyze` to refresh the workspace routes.

Example export body:

```json
{
  "format": "yolo_detection",
  "policy": "reviewed",
  "splits": ["train", "val"],
  "include_images": true
}
```

`splits: []` selects all splits. Bad/unsupported formats and empty export selections
return HTTP 400; stale previews and replacing a locked dataset return HTTP 409. Source paths are local
server paths, just like the existing image directory setting.

## Adding an adapter

Implement `read(root, source, labels, split) -> Dataset` and/or
`write(dataset, output) -> dict` as specified in `registry.py`. The writer returns
`image_paths`, `category_mapping`, and `warnings`; the service handles revision selection,
optional image copying, report writing and ZIP packaging. An optional `image_encoding: "jpeg"`
requests JPEG conversion (used by VOC). Output paths must remain
inside the export directory. Register a `Format` in `dataset_io/__init__.py` with
explicit supported tasks. The UI obtains options from the registry capability endpoint.

The implementation was checked against the [COCO data format](https://cocodataset.org/#format-data)
and [Ultralytics detection dataset specification](https://docs.ultralytics.com/datasets/detect/).
VOC directory/JPEG layout also follows the [Torchvision VOC loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/voc.py).

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

Tests cover cross-format review/reload/re-export, explicit empty human annotations,
source preservation, category remapping, Unicode/nested paths, split lists and filtering,
partial import reports, malformed rows, unsafe paths/YAML/XML, datasets over 200 images,
multi-root merge, preview/no-mutation, duplicate resolution, category mapping, stale-plan
rejection, idempotent commits, legacy append, preserved human/AI revisions, VOC pixel
conventions/flags/JPEG conversion and COCO → VOC → YOLO interoperability.

Optional browser acceptance (install a Playwright version compatible with your OS;
Ubuntu 20.04 uses 1.48.0):

```bash
pip install playwright==1.48.0
python -m playwright install chromium
VISIONREFINE_BROWSER_TESTS=1 pytest -q tests/test_dataset_io_browser.py
```

The browser test uses an isolated temporary project and real pointer events to preview
COCO import, relabel/save a box, download YOLO, reload, append two sources, map a category,
resolve a duplicate, verify preserved human edits/history, and download VOC.
