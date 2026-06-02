# Colab Sweep Summary

- Best run: `lr_5e-4`
- Best validation loss: `3.9764` at step `286`

| run | lr | context | layers | emb | dropout | best val | final val | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lr_5e-4 | 5e-04 | 64 | 2 | 128 | 0.1 | 3.9764 | 3.9764 | 532224 |
| lr_3e-4 | 3e-04 | 64 | 2 | 128 | 0.1 | 4.3681 | 4.3681 | 532224 |
| lr_1e-4 | 1e-04 | 64 | 2 | 128 | 0.1 | 5.2464 | 5.2464 | 532224 |

Generated plots:
- `val_loss_by_epoch.png`
- `train_val_loss_by_epoch.png`
