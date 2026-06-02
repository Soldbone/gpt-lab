# Colab Sweep Summary

- Best run: `dropout_0`
- Best validation loss: `3.0066` at step `3304`

| run | lr | context | layers | emb | dropout | best val | final val | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dropout_0 | 3e-04 | 64 | 2 | 128 | 0.0 | 3.0066 | 3.0066 | 532224 |
| lr_5e-4 | 5e-04 | 64 | 2 | 128 | 0.1 | 3.0098 | 3.0098 | 532224 |
| layers_4 | 3e-04 | 64 | 4 | 128 | 0.1 | 3.0255 | 3.0255 | 928000 |
| lr_3e-4 | 3e-04 | 64 | 2 | 128 | 0.1 | 3.1618 | 3.1618 | 532224 |
| ctx_128 | 3e-04 | 128 | 2 | 128 | 0.1 | 3.3771 | 3.3777 | 540416 |
| lr_1e-4 | 1e-04 | 64 | 2 | 128 | 0.1 | 3.5216 | 3.5216 | 532224 |

Generated plots:
- `val_loss_by_epoch.png`
- `train_val_loss_by_epoch.png`
