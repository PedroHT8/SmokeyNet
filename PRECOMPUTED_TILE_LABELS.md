# Extension: official precomputed tile labels

## Purpose

This extension supports the reproducible training of SmokeyNet on the original
paper split when the source contour masks are not available. It uses the
official file `data/label_stats/labels_stats_90overlap.pkl`, already included
in the upstream repository, instead of replacing missing annotations with
manually created bounding boxes or global labels.

The implementation does not change the SmokeyNet architecture, component
losses, or inference procedure. It changes only how tile-level supervision is
loaded during training.

## Origin and interpretation of the labels

The upstream helper notebook generated `labels_stats_90overlap.pkl` from the
official smoke masks using the same preprocessing geometry as the paper. Each
dictionary entry maps an image identifier to 45 smoke-pixel counts, one for
each tile. A tile is converted to a positive binary label when its count is
strictly greater than 250 pixels, matching the original training code.

The statistics are tied to this fixed geometry:

- resized image: `1392 x 1856` pixels;
- bottom crop: `1040` pixels high;
- tile size: `224 x 224` pixels;
- overlap between adjacent tiles: `20` pixels;
- grid: `5 x 9`, for a total of 45 tiles;
- temporal input: two frames in the paper-like experiment.

## Dataset coverage

The official training split contains 11,347 images from 144 fires:

| Group | Images |
| --- | ---: |
| Positive images in the original split | 5,721 |
| Positive images covered by tile statistics | 5,013 |
| Negative images retained | 5,626 |
| Positive images omitted for missing statistics | 708 |
| Effective training set | 10,639 |

At the 250-pixel threshold, the effective training set contains 12,832
positive tiles and 465,923 negative tiles. The resulting negative/positive
ratio is 36.31, which is consistent with the paper configuration
`bce_pos_weight=40`.

These counts can be reproduced without loading image data:

```bash
python scripts/validate_tile_label_stats.py \
  --stats data/label_stats/labels_stats_90overlap.pkl \
  --train-split data/final_split/train_images_final.txt
```

## Code changes

`src/main.py` adds the optional argument `--tile-label-stats-path` and passes it
to the data module. Existing commands and mask-based training remain unchanged
when the argument is omitted.

`src/dynamic_dataloader.py` implements the following behavior when the option
is enabled:

1. Load and validate the statistics dictionary once in the data module.
2. Require the exact paper geometry and reject random crop jitter.
3. Verify that every entry contains exactly 45 tile counts.
4. Remove positive training images that have no tile statistics.
5. Retain negative images and assign their tiles zero labels.
6. Threshold the precomputed counts at more than 250 smoke pixels.
7. Reverse the nine tile columns when horizontal flip augmentation is applied.
8. Reject incompatible object-detection and random tile-sampling modes.

Validation and test retain their global image labels. They must use
`--error-as-eval-loss`, as in the paper experiments, because complete
tile-level annotations are not available for these splits.

## Recommended paper-like command options

```bash
--tile-label-stats-path data/label_stats/labels_stats_90overlap.pkl \
--no-resize-crop-augment \
--error-as-eval-loss \
--series-length 2 \
--tile-size 224 \
--tile-overlap 20 \
--smoke-threshold 250 \
--bce-pos-weight 40
```

The complete Colab workflow is available at
`notebooks/SmokeyNet_PaperDataset_precomputed.ipynb`. It performs dataset
download, coverage checks, a data-loader smoke test, training, explicit best
checkpoint evaluation, and persistent checkpoint storage in Google Drive.

## Interrupted Colab sessions

The full experiment can exceed the duration or compute allowance of one Colab
session. The notebook therefore passes a persistent Google Drive directory via
`--checkpoint-dir`. Lightning updates `last.ckpt` there after every completed
epoch. When `RUN_MODE='full'` and `AUTO_RESUME=True`, a later session detects
that file and passes it through `--checkpoint-path`.

This is an exact Lightning resume: model weights, optimizer state, scheduler,
epoch number, callback state, and global step are restored. It must not be
confused with `--is-extra-training`, which loads weights but starts a new
optimization run. A disconnection can still lose the unfinished current epoch,
but all completed epochs remain in Drive.

The effective batch remains 32. The notebook uses batch size 1 with gradient
accumulation 32 on smaller GPUs and batch size 2 with accumulation 16 when at
least 30 GB of GPU memory is available.

## Methodological justification for the TFM

Using the official aggregate tile statistics is preferable to generating new
pseudo-labels because it preserves supervision derived from the annotations
used by the original authors. It also avoids treating unannotated positive
images as negative at tile level, a failure mode that can force the detector
toward degenerate all-positive or all-negative solutions.

This procedure should be described as a reconstruction of tile-level binary
supervision, not as a recovery of the original pixel masks. The counts allow
the same tile labels to be obtained for the fixed paper geometry, but they do
not permit visualization or segmentation of the exact smoke contour.

## Limitations

- The 708 omitted positive images cannot contribute tile-level supervision.
- Geometric crop or scale augmentation is incompatible with fixed tile counts.
- Two hundred and one covered positive images have no tile above the
  250-pixel threshold; they still contribute through the global image loss.
- Tile metrics on validation and test are not interpretable without complete
  tile annotations; image-level metrics are the primary evaluation target.
- Reproducing the data and training protocol does not guarantee identical
  paper results because runtime libraries, GPU kernels, random initialization,
  and unavailable original checkpoints may differ.
