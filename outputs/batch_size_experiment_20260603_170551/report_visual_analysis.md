# Batch Size Experiment Report

## Loss Graph Interpretation
The lowest validation loss was achieved by batch_size 4 with best_val_loss=4.0773.
- batch_size 4: best_epoch=10, best_val_loss=4.0773, final_val_loss=4.0773.
- batch_size 8: best_epoch=10, best_val_loss=4.1646, final_val_loss=4.1646.
- batch_size 16: best_epoch=10, best_val_loss=4.2240, final_val_loss=4.2240.

## Perplexity Graph Interpretation
The lowest test perplexity at the best validation epoch was batch_size 4 with test_ppl_at_best_epoch=59.3438.
- batch_size 4: best_val_ppl=58.9859, final_test_ppl=59.3438.
- batch_size 8: best_val_ppl=64.3675, final_test_ppl=64.9118.
- batch_size 16: best_val_ppl=68.3040, final_test_ppl=68.9680.

## Top-N Accuracy Interpretation
- batch_size 4: best-epoch test top-1/top-3/top-5=0.2272/0.3745/0.4476; final=0.2272/0.3745/0.4476.
- batch_size 8: best-epoch test top-1/top-3/top-5=0.2178/0.3618/0.4344; final=0.2178/0.3618/0.4344.
- batch_size 16: best-epoch test top-1/top-3/top-5=0.2131/0.3555/0.4266; final=0.2131/0.3555/0.4266.

## Best Epoch And Stability
The smallest final validation-train loss gap was batch_size 16 with absolute gap=0.1105.
- batch_size 4: final_loss_gap=0.1401, average_epoch_time=71.41 seconds.
- batch_size 8: final_loss_gap=0.1292, average_epoch_time=45.18 seconds.
- batch_size 16: final_loss_gap=0.1105, average_epoch_time=31.26 seconds.

## Training Time And Memory
The fastest total run was batch_size 16 (312.58 seconds), and the slowest was batch_size 4 (714.12 seconds).

## Generation Sample Interpretation
- input='.', batch_size=4: Automatic check found no strong immediate repetition.
- input='.', batch_size=8: Automatic check found no strong immediate repetition.
- input='.', batch_size=16: Automatic check found no strong immediate repetition.
- input='계피는 싫고 시나몬은 좋', batch_size=4: Automatic check found no strong immediate repetition.
- input='계피는 싫고 시나몬은 좋', batch_size=8: Automatic check found no strong immediate repetition.
- input='계피는 싫고 시나몬은 좋', batch_size=16: Automatic check found no strong immediate repetition.
- input='"남자 주인공 몸에 살만 붙으면,그대로-제2의""조춘""선생님이다...나무아미타불~관세음 ', batch_size=4: Automatic check found no strong immediate repetition.
- input='"남자 주인공 몸에 살만 붙으면,그대로-제2의""조춘""선생님이다...나무아미타불~관세음 ', batch_size=8: Automatic check found no strong immediate repetition.
- input='"남자 주인공 몸에 살만 붙으면,그대로-제2의""조춘""선생님이다...나무아미타불~관세음 ', batch_size=16: Automatic check found no strong immediate repetition.

## Recommendation
Based on validation loss and test perplexity, batch_size 4 is the primary recommendation.
