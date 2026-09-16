# VisionRefine

VisionRefine is a local-first, AI-assisted multimodal annotation workflow. It
inspects a dataset, detects existing coarse annotations, chooses how each image
should be presented to a bounded-resolution model, and tracks AI and human
review stages without silently overwriting verified annotations.

## Current prototype

- Tasks: detection, instance segmentation, visual grounding, captioning, VQA,
  OCR, and image classification.
- Inputs: a local image directory and an optional coarse annotation file.
- Resolution routing: direct, resized whole image, overlapping tiles, or crops
  around coarse annotations.
- Workflow: initial/refine -> gap check -> human review -> AI final audit.
- OpenAI-compatible VLM adapter with model discovery and a single-tile pilot.
- Gigapixel annotation workspace with one zoomable canvas and on-demand
  original-resolution crops.
- Bounding-box creation, selection, movement, edge/corner resizing, deletion,
  mouse-centered zoom, and panning.
- Separate AI suggestion and human-reviewed revisions with reload persistence.
- Durable project manifests under `workspace/projects/`.

The next implementation stage is a per-image candidate queue: existing coarse
annotations will be refined by crops, while unlabelled datasets will receive
detector candidates before selective VLM review. This avoids sending every tile
of a gigapixel image to a VLM.

## Run locally

```bash
cd /Users/soleilor/Desktop/Github/VisionRefine
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
visionrefine --port 8020
```

Open <http://127.0.0.1:8020>.

## Bounding-box review

1. Open a project and choose **Open annotation workspace**.
2. Use the wheel to zoom around the pointer and right-drag the image to pan.
3. Drag on empty space to add a box. Click a box to select it, then drag the box
   or one of its eight handles. Press Delete to remove it.
4. Save the human review. Reloading the image reads the saved human revision.

## Safety model

AI output is stored as a suggestion revision. Human-verified annotations are a
separate revision and are never silently replaced by a later AI pass.

## License

Apache-2.0.
