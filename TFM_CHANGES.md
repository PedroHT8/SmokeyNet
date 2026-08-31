# Modifications with respect to the original SmokeyNet implementation

## 1. Reference implementation

This repository is based on the original SmokeyNet implementation:

https://gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection

associated with the work:

A. Dewangan et al., *Smoke Detection for Early Wildfire Using Deep Learning*, Remote Sensing, 2022.

The modifications described below were introduced to reproduce and evaluate SmokeyNet in a modern Google Colab environment and to recover tile-level supervision using the precomputed statistics available for the original dataset.

Unless otherwise stated, these modifications do not alter the architecture of the ViT-based SmokeyNet baseline.

## 2. Scope of the modifications

The original SmokeyNet architecture is preserved:

* CNN-based feature extraction at tile level.
* LSTM-based temporal modelling.
* Spatial Vision Transformer for spatial aggregation.
* Original tile-level and image-level prediction structure.

The changes introduced in this fork are limited to:

1. Compatibility with current versions of PyTorch, PyTorch Lightning and TorchMetrics.
2. Minor corrections required for execution and evaluation.
3. Support for precomputed tile-level supervision.
4. Removal of machine-specific data paths.
5. Persistent checkpoint storage and training resumption.

## 3. PyTorch compatibility

### `src/util_fns.py`

The original implementation imported:

```python
from torch._six import string_classes
```

`torch._six` was an internal PyTorch compatibility module that is no longer available in recent PyTorch releases.

It was replaced with:

```python
string_classes = (str, bytes)
```

This change only provides compatibility with current PyTorch versions and does not modify the behaviour of the model.

## 4. TorchMetrics compatibility

### `src/lightning_module.py`

The original repository used an older TorchMetrics API based on arguments such as:

```python
multiclass=False
mdmc_average=...
```

The metrics were migrated to the current binary classification API:

```python
torchmetrics.Accuracy(task="binary")
torchmetrics.Precision(task="binary")
torchmetrics.Recall(task="binary")
torchmetrics.F1Score(task="binary")
```

The evaluated quantities remain accuracy, precision, recall and F1-score.

## 5. PyTorch Lightning compatibility

### `src/lightning_module.py`

The original implementation used:

```python
test_epoch_end(test_step_outputs)
```

which is no longer supported by recent PyTorch Lightning versions.

Test outputs are now accumulated during `test_step()` and processed using:

```python
on_test_epoch_end()
```

The accumulated outputs are cleared after evaluation.

This preserves the original test-time processing, including the computation of additional wildfire evaluation metrics such as time to detection.

### `src/main.py`

Several Trainer arguments were migrated to the current PyTorch Lightning API.

GPU selection:

```python
gpus=1
```

was replaced by:

```python
accelerator="gpu"
devices=1
```

Mixed-precision configuration was updated to:

```python
precision="16-mixed"
```

when 16-bit training is enabled.

Checkpoint resumption was migrated from the deprecated Trainer argument to:

```python
trainer.fit(..., ckpt_path=...)
```

and checkpoint-based test evaluation uses:

```python
trainer.test(..., ckpt_path=...)
```

Stochastic Weight Averaging, originally exposed through the Trainer configuration, is now implemented using the corresponding PyTorch Lightning callback.

The calculation of `log_every_n_steps` was also changed to guarantee a positive integer value.

## 6. General corrections

### Train/test split argument types

In the original `src/main.py`, the train and test split proportions were declared as integers while their default values were fractional:

```python
type=int, default=0.7
type=int, default=0.15
```

These arguments were corrected to:

```python
type=float
```

### Image loading

The original dataloader contained a fallback path specific to the infrastructure used by the original authors.

If an image was not found under the user-provided data path, the implementation attempted to load it from a hard-coded path under `/userdata/`.

This machine-specific fallback was removed.

Images are now loaded exclusively from the configured `raw_data_path`, and a `FileNotFoundError` is raised when an expected image is unavailable.

### Image-level probabilities

In `src/main_model.py`, `image_probs` is explicitly initialised.

When image-level predictions are derived from tile predictions rather than from a dedicated image output, the image probability is computed as:

```python
image_probs = tile_probs.max(dim=1).values
```

This provides a continuous image-level score consistent with the decision rule in which the presence of a positive tile makes the image positive.

It also enables the calculation of probability-based evaluation metrics.

## 7. Precomputed tile-level supervision

### Motivation

SmokeyNet uses tile-level supervision during training.

To reproduce this training procedure without relying on unavailable image-mask files, this fork supports loading precomputed smoke-pixel statistics for the tiles associated with each image.

A new command-line argument was added:

```text
--tile-label-stats-path
```

The corresponding pickle file is loaded as a dictionary indexed by image name.

### Tile geometry

The precomputed statistics correspond to the geometry used by the original SmokeyNet configuration:

* resized image: 1392 × 1856 pixels;
* crop height: 1040 pixels;
* tile dimensions: 224 × 224 pixels;
* tile overlap: 20 pixels.

The resulting spatial grid contains 45 tiles.

The dataloader verifies that the configured geometry matches this geometry before allowing the precomputed labels to be used.

### Tile-label generation

Each entry contains the smoke-pixel count associated with every tile.

The existing `smoke_threshold` configuration is applied to obtain binary tile labels:

```python
tiled_labels = (tile_counts > self.smoke_threshold).astype(float)
```

This produces the tile-level ground truth used by the original classification pipeline.

### Validation

The implementation verifies that every statistics entry contains the expected number of tiles.

Precomputed tile supervision is rejected for configurations that are incompatible with these labels, including:

* object-detection mode;
* random tile sampling;
* geometric crop augmentation incompatible with the fixed tile grid.

### Positive images without tile statistics

A positive training image for which no tile-level statistics are available cannot be assigned reliable local supervision.

Such images are therefore removed from the training split when the precomputed-label workflow is enabled.

Negative images can still be used because their tiles can consistently be considered negative.

### Horizontal flipping

When horizontal image augmentation is performed, the spatial arrangement of the precomputed tile labels is transformed accordingly.

The tile matrix is reshaped into its two-dimensional grid, its columns are reversed and it is flattened again.

This ensures that image augmentation and tile-level supervision remain spatially aligned.

## 8. Persistent checkpoints and training resumption

### `src/main.py`

A new optional argument was introduced:

```text
--checkpoint-dir
```

It allows the `ModelCheckpoint` callback to write the best and last checkpoints directly to a persistent directory.

This adaptation is useful for long-running experiments in temporary environments such as Google Colab.

Training can subsequently be resumed from a saved checkpoint through:

```text
--checkpoint-path
```

using the current PyTorch Lightning `ckpt_path` interface.

Checkpoint-based resumption restores the model and training state managed by PyTorch Lightning.

## 9. Architectural impact

For the ViT baseline used in the experiments, the changes described in this document do **not** replace or redesign the principal architectural components of SmokeyNet.

In particular, the following components remain those of the original implementation:

* tile-level CNN feature extraction;
* temporal LSTM processing;
* Spatial Vision Transformer;
* tile/image prediction hierarchy.

The principal methodological modification is therefore related to how tile-level training supervision is recovered and supplied to the original model, rather than to a redesign of the ViT-based architecture.

A separate architectural extension replacing the Spatial Vision Transformer with a Mamba-based module can be documented independently if incorporated into this repository.

## 10. Summary of modified source files

| File                        | Purpose of modification                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------------------- |
| `.gitignore`                | Prevent local Python caches and virtual environments from being versioned                       |
| `src/util_fns.py`           | Compatibility with current PyTorch versions                                                     |
| `src/lightning_module.py`   | TorchMetrics and PyTorch Lightning compatibility                                                |
| `src/main.py`               | Modern Lightning API, tile-label configuration, argument corrections and persistent checkpoints |
| `src/main_model.py`         | Reliable image-level probability handling                                                       |
| `src/dynamic_dataloader.py` | Portable image loading and precomputed tile-level supervision                                   |

These changes provide a reproducible modern implementation while preserving the architecture of the original ViT-based SmokeyNet baseline.
