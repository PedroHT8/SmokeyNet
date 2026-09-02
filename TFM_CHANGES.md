# Modifications with respect to the original SmokeyNet implementation

## 1. Reference implementation

This repository is based on the original SmokeyNet implementation:

https://gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection

associated with the work:

A. Dewangan et al., *Smoke Detection for Early Wildfire Using Deep Learning*, Remote Sensing, 2022.

The modifications described below were introduced to reproduce and evaluate SmokeyNet in a modern Google Colab environment and to recover tile-level supervision using the precomputed statistics available for the original dataset.

Two TFM branches are maintained:

* `tfm/smokeynet-vit`: reproducible ViT-based SmokeyNet baseline.
* `tfm/smokeynet-mamba`: extension of the previous branch replacing the Spatial Vision Transformer with a Mamba-based spatial module.

Unless otherwise stated, the compatibility and data-processing modifications described in Sections 2–8 do not alter the architecture of the ViT-based SmokeyNet baseline.

## 2. Scope of the modifications

The original SmokeyNet baseline architecture is preserved in `tfm/smokeynet-vit`:

* CNN-based feature extraction at tile level.
* LSTM-based temporal modelling.
* Spatial Vision Transformer for spatial aggregation.
* Original tile-level and image-level prediction structure.

The common changes introduced in this fork are limited to:

1. Compatibility with current versions of PyTorch, PyTorch Lightning and TorchMetrics.
2. Minor corrections required for execution and evaluation.
3. Support for precomputed tile-level supervision.
4. Removal of machine-specific data paths.
5. Persistent checkpoint storage and training resumption.

The branch `tfm/smokeynet-mamba` additionally introduces one architectural modification: replacement of `TileToTileImage_SpatialViT` with the Mamba-based `TileToTileImage_SpatialVim` module described in Section 10.

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

## 9. Architectural impact of the ViT baseline

For the ViT baseline used in the experiments, the changes described in Sections 2–8 do **not** replace or redesign the principal architectural components of SmokeyNet.

In particular, the following components remain those of the original implementation:

* tile-level CNN feature extraction;
* temporal LSTM processing;
* Spatial Vision Transformer;
* tile/image prediction hierarchy.

The principal methodological modification of `tfm/smokeynet-vit` is therefore related to how tile-level training supervision is recovered and supplied to the original model, rather than to a redesign of the architecture.

## 10. Mamba-based architectural extension

### 10.1 Motivation

The branch `tfm/smokeynet-mamba` implements an architectural variant in which the original spatial aggregation module:

```text
TileToTileImage_SpatialViT
```

is replaced with:

```text
TileToTileImage_SpatialVim
```

The preceding components remain unchanged:

```text
RawToTile_MobileNet
        ↓
TileToTile_LSTM
        ↓
TileToTileImage_SpatialVim
```

The purpose of this variant is to compare the original Transformer-based spatial processing with a state-space-model-based alternative while preserving the rest of the SmokeyNet processing pipeline.

### 10.2 Input and output contract

The LSTM produces tile embeddings with shape:

```text
[B, 45, T, 960]
```

where:

* `B` is the batch size;
* `45` corresponds to the 5 × 9 spatial tile grid;
* `T` is the temporal sequence length;
* `960` is the embedding dimension produced by the MobileNet/LSTM pipeline.

`TileToTileImage_SpatialVim` preserves the same external prediction structure expected by SmokeyNet:

```text
tile_outputs  [B, 45, T]
image_outputs [B, T]
```

This allows the existing losses, evaluation pipeline and image/tile prediction hierarchy to remain unchanged.

### 10.3 Projection to the spatial latent representation

The 960-dimensional LSTM embeddings are projected to 516 dimensions:

```python
self.input_projection = nn.Linear(
    tile_embedding_size,
    vim_d_model,
)
```

with:

```text
tile_embedding_size = 960
vim_d_model = 516
```

The value 516 was selected to preserve the latent dimensionality used by the original `SpatialViT`, whose `ViTConfig` uses:

```python
hidden_size=516
```

and whose prediction head is:

```python
TileEmbeddingsToOutput(516)
```

The Mamba-based variant therefore retains a 516-dimensional spatial representation before the common SmokeyNet prediction head.

### 10.4 Spatial sequence construction

Spatial processing is performed independently for every temporal position.

The embeddings are reorganised from:

```text
[B, N, T, D]
```

to:

```text
[B*T, N, D]
```

so that the 45 tiles form the sequence processed by the spatial module.

The tiles retain their raster ordering.

A learned CLS token is inserted in the centre of the sequence:

```text
22 tiles + CLS + 23 tiles
```

Learned absolute positional embeddings are then added to the 45 tile tokens and the CLS token.

### 10.5 Mamba blocks and bidirectional processing

The Mamba mixer used inside the module is imported from the official `mamba-ssm` package:

```python
from mamba_ssm import Mamba
```

The implementation does not reproduce the internal Selective State Space Model operations.

Each `_VimMambaBlock` follows a non-fused:

```text
Add -> Norm -> Mixer
```

data flow and uses PyTorch `LayerNorm`.

Four Mamba layers are instantiated and consumed as two bidirectional pairs:

```text
Pair 1:
    layer 0 -> forward spatial order
    layer 1 -> reversed spatial order

Pair 2:
    layer 2 -> forward spatial order
    layer 3 -> reversed spatial order
```

For the backward branch, the token sequence is reversed before the Mamba layer and restored to its original order afterwards.

The representations obtained from both directions are then combined:

```text
forward + reverse(backward)
```

The same bidirectional combination is applied to the residual stream.

This processing strategy is based on the bidirectional sequence-processing approach used by Vision Mamba (Vim).

### 10.6 Global and tile-level predictions

After the final residual addition and `LayerNorm`, the central CLS token is used as the global image representation.

The remaining 45 contextualised tokens retain the original tile ordering and are used as local spatial representations.

The embeddings are reconstructed following the same convention as `SpatialViT`:

```text
[CLS, tile1, ..., tile45]
```

and passed through:

```python
TileEmbeddingsToOutput(516)
```

The first output corresponds to the image prediction and the remaining 45 outputs correspond to tile predictions.

### 10.7 SpatialVim configuration

The implementation used in the TFM experiments uses:

```text
d_model = 516
d_state = 16
d_conv = 4
expand = 2
depth = 4
pos_dropout = 0.0
drop_path = 0.0
residual_in_fp32 = True
```

With `depth=4`, the module contains two bidirectional forward/backward Mamba pairs.

### 10.8 Provenance of the implementation

The Mamba-based extension combines three sources:

**Original SmokeyNet**

The following components and interfaces are retained from the original implementation:

* MobileNet tile feature extraction;
* LSTM temporal processing;
* 5 × 9 tile organisation;
* tile/image prediction hierarchy;
* `TileEmbeddingsToOutput`.

**Official Mamba implementation**

The core Mamba mixer is provided by the official `mamba-ssm` package.

The Selective State Space Model itself is therefore not reimplemented in this repository.

**Vision Mamba-inspired adaptation**

The following design elements are based on the Vision Mamba (Vim) approach:

* Add → Norm → Mixer block organisation;
* separate residual stream;
* central CLS token;
* positional embeddings;
* bidirectional forward/backward processing.

The adaptation connecting these elements to the SmokeyNet LSTM embeddings and reconstructing the original SmokeyNet tile/image outputs was developed specifically for this TFM.

The resulting module should therefore be described as a **Mamba-based spatial extension of SmokeyNet inspired by Vision Mamba**, rather than as a complete implementation of Vision Mamba or VMamba.

## 11. Validation of the Mamba integration

The branch `tfm/smokeynet-mamba` was first verified using a direct forward-pass test of `TileToTileImage_SpatialVim`.

For an input tensor with shape:

```text
[1, 45, 2, 960]
```

the module produced:

```text
tile_outputs:  [1, 45, 2]
image_outputs: [1, 2]
```

The complete integration was subsequently tested using real FIgLib images and the same geometry used for the ViT baseline:

* resize: 1392 × 1856;
* crop height: 1040;
* tile size: 224 × 224;
* tile overlap: 20;
* spatial grid: 5 × 9 = 45 tiles;
* sequence length: 2.

The smoke test included:

* `RawToTile_MobileNet`;
* `TileToTile_LSTM`;
* `TileToTileImage_SpatialVim`;
* precomputed tile-level labels;
* mixed-precision training;
* one complete training epoch;
* validation;
* test evaluation;
* checkpoint generation.

The validated implementation corresponds to commit:

```text
1576abf22d84a09bc27ba4f339bf27d527fc9f40
```

The execution completed with exit code `0`, generated `last.ckpt` successfully and left the cloned repository without local source-code modifications.

The numerical metrics obtained in this smoke test are not intended as performance results because of the deliberately reduced dataset and single training epoch. Its purpose was exclusively to verify the correct end-to-end integration of the new spatial module.

## 12. Summary of modified source files

| File                        | Purpose of modification                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------- |
| `.gitignore`                | Prevent local Python caches and virtual environments from being versioned                               |
| `src/util_fns.py`           | Compatibility with current PyTorch versions                                                             |
| `src/lightning_module.py`   | TorchMetrics and PyTorch Lightning compatibility                                                        |
| `src/main.py`               | Modern Lightning API, tile-label configuration, argument corrections and persistent checkpoints         |
| `src/main_model.py`         | Reliable image-level probability handling                                                               |
| `src/dynamic_dataloader.py` | Portable image loading and precomputed tile-level supervision                                           |
| `src/model_components.py`   | Adds the optional Mamba dependency and `TileToTileImage_SpatialVim` in the `tfm/smokeynet-mamba` branch |

The branch `tfm/smokeynet-vit` provides the reproducible modern implementation of the original ViT-based SmokeyNet architecture.

The branch `tfm/smokeynet-mamba` extends that baseline with the Mamba-based spatial module while preserving the MobileNet, LSTM, supervision, loss and evaluation pipeline used by the ViT baseline.
