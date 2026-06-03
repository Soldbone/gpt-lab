# Batch Size Experiment Report

## Loss Graph Interpretation
The lowest validation loss was achieved by batch_size 2 with best_val_loss=5.6123.
- batch_size 2: best_epoch=1, best_val_loss=5.6123, final_val_loss=5.6123.
- batch_size 4: best_epoch=1, best_val_loss=5.6566, final_val_loss=5.6566.

## Perplexity Graph Interpretation
The lowest test perplexity at the best validation epoch was batch_size 2 with test_ppl_at_best_epoch=287.1975.
- batch_size 2: best_val_ppl=273.7612, final_test_ppl=287.1975.
- batch_size 4: best_val_ppl=286.1653, final_test_ppl=313.4067.

## Top-N Accuracy Interpretation
- batch_size 2: best-epoch test top-1/top-3/top-5=0.0156/0.0312/0.0312; final=0.0156/0.0312/0.0312.
- batch_size 4: best-epoch test top-1/top-3/top-5=0.0078/0.0156/0.0234; final=0.0078/0.0156/0.0234.

## Best Epoch And Stability
The smallest final validation-train loss gap was batch_size 2 with absolute gap=0.0579.
- batch_size 2: final_loss_gap=-0.0579, average_epoch_time=0.49 seconds.
- batch_size 4: final_loss_gap=-0.1400, average_epoch_time=0.30 seconds.

## Training Time And Memory
The fastest total run was batch_size 4 (0.30 seconds), and the slowest was batch_size 2 (0.49 seconds).
GPU memory metrics are N/A because CUDA was not available or memory tracking was not supported.

## Generation Sample Interpretation
- input='.', batch_size=2: Automatic check found no strong immediate repetition.
- input='.', batch_size=4: Automatic check found no strong immediate repetition.
- input='계피는 싫고 시나몬은 좋', batch_size=2: Automatic check found no strong immediate repetition.
- input='계피는 싫고 시나몬은 좋', batch_size=4: Automatic check found no strong immediate repetition.
- input='"남자 주인공 몸에 살만 붙으면,그대로-제2의""조춘""선생님이다...나무아미타불~관세음 ', batch_size=2: Automatic check found no strong immediate repetition.
- input='"남자 주인공 몸에 살만 붙으면,그대로-제2의""조춘""선생님이다...나무아미타불~관세음 ', batch_size=4: Automatic check found no strong immediate repetition.

## Recommendation
Based on validation loss and test perplexity, batch_size 2 is the primary recommendation.
