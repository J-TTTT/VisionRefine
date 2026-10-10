"""Task-scoped, versioned adapters. Register here, never in the UI."""
from .coco import CocoDetection
from .coco_segmentation import CocoSegmentation
from .common import ImagesImporter
from .native import NativeDataset
from .registry import Format, FormatRegistry
from .yolo import YoloDetection
from .voc import VocDetection
from .tool_formats import CvatDetection, LabelmeDetection, LabelStudioDetection

registry = FormatRegistry()
registry.register(Format("images", "仅图片", ("detection", "instance_segmentation", "grounding", "captioning", "vqa", "ocr", "classification"), ImagesImporter(),
    geometries=(), input=dict(kind="none", required=False, labels=True, hint="递归扫描本地图片；非检测任务目前仅建立图像清单。")))
registry.register(Format("coco_segmentation", "COCO Instance Segmentation", ("instance_segmentation",),
    CocoSegmentation(), CocoSegmentation(), version="COCO polygon + compressed RLE", geometries=("polygon", "mask"),
    attributes=("iscrowd",), preserves_splits=True, status="tested",
    input=dict(kind="file", required=True, labels=False, hint="选择 COCO instances JSON；图片根目录对应 file_name。支持多边形与压缩/未压缩 RLE。",
               placeholder="/path/to/instances.json"),
    limitations=("复杂或越界多边形导入为像素掩码；超过 4096 块的掩码无法导入。", "整图 RLE 不支持超过 2³²−1 像素的图像。"),
    reference="https://github.com/cocodataset/cocoapi"))


def detection(id, title, adapter, *, version, attributes=(), confidence=False, splits=False, kind="file", required=True,
              hint="", placeholder="", reference="", limitations=(), status="tested"):
    registry.register(Format(id, title, ("detection",), adapter, adapter, version=version, attributes=attributes,
        preserves_confidence=confidence, preserves_splits=splits, status=status,
        input=dict(kind=kind, required=required, labels=id in {"voc_detection", "labelme_detection", "label_studio_detection"},
                   split_from_source=id in {"yolo_detection", "visionrefine"}, hint=hint, placeholder=placeholder),
        limitations=limitations or ("仅普通矩形检测框；不支持旋转框、多边形、视频或其他任务。",), reference=reference))


detection("coco_detection", "COCO Detection", CocoDetection(), version="Detection 1.0", attributes=("iscrowd",), splits=True,
    hint="file_name 相对于图片根目录；导出含各划分独立 JSON。", placeholder="/path/to/annotations.json", reference="https://cocodataset.org/#format-data")
detection("yolo_detection", "YOLO Detection", YoloDetection(), version="Ultralytics YAML + 5-column TXT", splits=True,
    hint="根目录包含 images/ 和 labels/；data.yaml 指定划分。", placeholder="/path/to/data.yaml", reference="https://docs.ultralytics.com/datasets/detect/")
detection("voc_detection", "Pascal VOC Detection", VocDetection(), version="VOC XML", attributes=("difficult", "truncated", "occluded", "pose"), splits=True,
    kind="directory", required=False, hint="根目录包含 JPEGImages、Annotations、ImageSets/Main。", placeholder="可选，默认根目录/Annotations")
detection("cvat_detection", "CVAT for images · Detection", CvatDetection(), version="XML 1.1", attributes=("*",), required=False,
    hint="图片根目录对应 XML 的 image name；标注文件为 annotations.xml。", placeholder="/path/to/annotations.xml", reference="https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/")
detection("label_studio_detection", "Label Studio · RectangleLabels", LabelStudioDetection(), version="JSON RectangleLabels v1", required=False,
    hint="根目录绑定 data.image 相对路径；只接收单版本、无旋转的 RectangleLabels。", placeholder="/path/to/annotations.json", reference="https://labelstud.io/guide/export")
detection("labelme_detection", "Labelme · Rectangle", LabelmeDetection(), version="wkentaro JSON 5.x", attributes=("flags", "group_id", "description"), kind="file_or_directory", required=False,
    hint="根目录包含图片；标注输入为 JSON 文件或目录，留空扫描根目录。", placeholder="/path/to/labelme-directory", reference="https://github.com/wkentaro/labelme")
detection("visionrefine", "VisionRefine 原生数据集快照", NativeDataset(), version="1.0", attributes=("*",), confidence=True, splits=True, required=False,
    hint="选择解压根目录与 dataset.json；也可指定原生 ZIP，系统解压到独立缓存。", placeholder="/path/to/dataset.json 或 native.zip",
    limitations=("检测快照，不是完整项目备份。默认不自动信任外部审核状态。",))
