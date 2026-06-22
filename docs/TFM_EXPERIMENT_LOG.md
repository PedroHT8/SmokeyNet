# TFM experiment log

This document records reproducible experiment results and their interpretation.
It is intended to support the methodology and results chapters of the TFM.

## Precomputed-label diagnostic run

Date: 2026-06-19

### Configuration

- Experiment: `smokeynet_paper_precomputed_diagnostic`
- Architecture: ResNet34 + LSTM + SpatialViT
- Epochs: 2
- Effective training images: 10,639
- Tile-label source: `labels_stats_90overlap.pkl`
- Series length: 2
- Effective batch size: 32 (`batch_size=1`, gradient accumulation 32)
- Tile positive weight: 40
- Image positive weight: 5
- Fixed paper geometry: resize 1392 x 1856, bottom crop 1040,
  224-pixel tiles, 20-pixel overlap

### Training evolution

| Metric | Epoch 0 | Epoch 1 |
| --- | ---: | ---: |
| Total training loss | 8.929152 | 5.562450 |
| Tile loss 0 | 1.963765 | 1.481019 |
| Tile loss 1 | 1.991615 | 1.935420 |
| Tile loss 2 | 1.156317 | 0.762731 |
| Image loss 3 | 3.817459 | 1.383278 |
| Training tile F1 | 0.050860 | 0.104084 |
| Training image accuracy | 0.494125 | 0.471191 |
| Training image precision | 0.473210 | 0.471191 |
| Training image recall | 0.650110 | 1.000000 |
| Validation image loss 3 | 1.371027 | 1.356627 |
| Validation image accuracy | 0.505930 | 0.505930 |
| Validation image F1 | 0.671917 | 0.671917 |

All four supervised losses decreased. Training tile F1 doubled, and validation
image BCE also decreased slightly. These trends show that gradients propagate
through the complete architecture and that the precomputed tile labels provide
a learnable signal.

### Best-checkpoint test result

The checkpoint callback selected `epoch=0-step=333.ckpt` because the monitored
`val/loss` is the image error rate at a fixed 0.5 threshold. That metric was
identical in both epochs.

| Test metric | Value |
| --- | ---: |
| Image accuracy | 0.505943 |
| Image precision | 0.505943 |
| Image recall | 1.000000 |
| Image F1 | 0.671928 |
| Negative accuracy | 0.000000 |
| Positive accuracy | 1.000000 |

At the fixed 0.5 threshold, the diagnostic checkpoint classified every test
image as positive. Its accuracy therefore matches the positive prevalence and
its apparently moderate F1 must not be interpreted as useful discrimination.
This is an early-training class collapse, not a successful detector.

Test tile F1, precision, and recall are not interpretable because validation
and test do not provide complete tile annotations. Image-level metrics are the
valid evaluation target for those splits.

### Decision

The diagnostic run validates the data and optimization pipeline but not final
detection performance. A full 25-epoch paper-like run is justified because the
component losses and tile F1 improve. Final analysis must compare the best and
last checkpoints and calibrate the image threshold on validation data before
reporting test performance. This avoids selecting an early checkpoint solely
because the fixed-threshold validation error is temporarily flat.

Because the estimated full training time on a T4 is 26-28 hours, the Colab
workflow stores `last.ckpt` directly in Google Drive after every epoch and
resumes it exactly across runtime sessions. Session interruption is therefore
treated as an infrastructure constraint rather than as a reason to shorten the
experimental protocol.
