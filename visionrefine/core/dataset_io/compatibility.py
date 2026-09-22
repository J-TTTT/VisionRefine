"""Preflight rules describe what the target format itself can represent."""
from collections import Counter


def issue(code, message, count=1, severity="conversion", examples=None):
    return dict(code=code, severity=severity, message=message, count=count,
                requires_ack=severity in {"loss", "conversion"}, examples=(examples or [])[:5])


def inspect_export(adapter, dataset, include_images):
    from .portable import redact
    result, omitted, unsupported = [], Counter(), Counter()
    metadata = [dataset.provenance, dataset.report.model_dump()]
    metadata.extend(c.provenance for c in dataset.categories)
    for image in dataset.images:
        metadata.append(image.provenance)
        for obj in image.objects:
            metadata.extend((obj.attributes, obj.provenance))
    if any(redact(value) != value for value in metadata):
        result.append(issue("private_metadata_removed", "为保护隐私，来源元数据中的本机绝对路径、密钥与连接配置将从所有导出文件及附加清单中移除。", severity="loss"))
    if dataset.report.issues:
        result.append(issue("prior_import_issues", "原始导入存在提示或错误；本次导出不能恢复之前被跳过的对象/字段，请核对导入报告。", len(dataset.report.issues), "info"))
    confidence = 0
    for image in dataset.images:
        for obj in image.objects:
            if obj.kind not in adapter.geometries:
                unsupported[obj.kind] += 1
            if "*" not in adapter.attributes:
                omitted.update(key for key in obj.attributes if key not in adapter.attributes)
            confidence += obj.confidence is not None
    if unsupported:
        result.append(issue("unsupported_geometry", "目标格式不能表示这些几何类型；本期不自动丢弃对象。", sum(unsupported.values()), "blocking", list(unsupported)))
    if omitted:
        result.append(issue("attributes_sidecar", "这些属性不能写入目标标注，只保留在 VisionRefine 附加清单中。", sum(omitted.values()), "loss", list(omitted)))
    if confidence and not adapter.preserves_confidence:
        result.append(issue("confidence_sidecar", "逐对象置信度只保留在附加清单中。", confidence, "loss"))
    if not adapter.preserves_splits and any(i.split != "unspecified" for i in dataset.images):
        result.append(issue("splits_sidecar", "目标标注不包含标准划分字段；划分保留在附加清单中。", severity="loss"))
    if adapter.id == "yolo_detection":
        count = sum(i.split == "unspecified" for i in dataset.images)
        if count:
            result.append(issue("unspecified_to_train", "YOLO 将未指定划分映射到 train。", count))
    if adapter.id == "voc_detection":
        bad = sum(any(o.attributes.get(k, 0) not in (0, 1) for k in ("difficult", "truncated", "occluded"))
                  or not isinstance(o.attributes.get("pose", "Unspecified"), str)
                  for image in dataset.images for o in image.objects)
        if bad:
            result.append(issue("invalid_voc_attributes", "VOC 的 difficult/truncated/occluded 必须是 0/1，pose 必须是文本。", bad, "blocking"))
        count = sum(any(v != int(v) for v in obj.bbox) for image in dataset.images for obj in image.objects)
        if count:
            result.append(issue("integer_boxes", "VOC 小数边界向外取整为整数像素。", count))
        count = sum(i.provenance.get("image_encoding") != "JPEG" if i.provenance.get("image_encoding")
                    else not i.path.lower().endswith((".jpg", ".jpeg")) for i in dataset.images)
        if count:
            result.append(issue("jpeg_conversion", "VOC 图片需为 JPEG；包含图片时转换为 RGB JPEG，可能有损。仅标注包需自行转换图片。", count))
    if not include_images:
        result.append(issue("images_omitted", "仅标注包需要按路径映射补齐图像，不能独立打开。", severity="info"))
    if adapter.id != "visionrefine":
        result.append(issue("workflow_sidecar", "来源与 VisionRefine 审核修订信息保存在附加清单，不等同于目标工具的审核状态。", severity="info"))
        oriented = sum(i.provenance.get("exif_orientation", 1) != 1 for i in dataset.images)
        if oriented:
            result.append(issue("exif_orientation", "图片带 EXIF 旋转/翻转信息，目标工具可能自动转向导致框错位；请先统一图像与标注坐标方向。本期不自动变换。", oriented, "blocking"))
    if adapter.id in {"voc_detection", "cvat_detection", "label_studio_detection"}:
        def valid_xml(value):
            return all(c in "\t\n\r" or 0x20 <= ord(c) <= 0xd7ff or 0xe000 <= ord(c) <= 0xfffd or ord(c) >= 0x10000 for c in value)
        texts = [c.name for c in dataset.categories]
        if adapter.id == "cvat_detection":
            texts.extend(k for image in dataset.images for o in image.objects for k in o.attributes)
            texts.extend(v for image in dataset.images for o in image.objects for v in o.attributes.values() if isinstance(v, str))
        if adapter.id == "voc_detection":
            texts.extend(o.attributes.get("pose", "Unspecified") for image in dataset.images for o in image.objects
                         if isinstance(o.attributes.get("pose", "Unspecified"), str))
        bad = sum(not valid_xml(t) for t in texts)
        if adapter.id == "voc_detection":
            bad += sum(any(ord(c) < 32 for c in category.name) for category in dataset.categories)
        if bad:
            result.append(issue("invalid_xml_text", "类别或属性含目标 XML / 类别清单不允许的控制字符。", bad, "blocking"))
    return result
