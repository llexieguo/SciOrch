# Datasets

SciOrch stores the prompt-based MAS baseline data used by the local
runner under `maslab/datasets/data`.

The combined benchmark JSON files are:

```text
maslab/datasets/data/train_combined.json
maslab/datasets/data/test_combined.json
```

`train_combined.json` and `test_combined.json` are copied from the reconstructed
train/test split and replace the earlier broken `data/test_combined.json`.

Images are stored under:

```text
maslab/datasets/data/images
```

The prompt-based MAS baseline runner expects a combined benchmark JSON via
`--input`, for example:

```bash
python -u maslab/inference.py \
  --method llm_debate \
  --input maslab/datasets/data/test_combined.json
```

Dataset loading and multimodal formatting are implemented in:

- `maslab.datasets.loader.load_combined_dataset`
- `maslab.utils.formatting.format_query`
- `maslab.utils.images.load_sample_images_with_warnings`
