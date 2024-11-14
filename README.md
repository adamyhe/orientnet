# ORIENTNET

For predicting/attributing sequence contribution to promoter orientation.

## Attribution

```bash
python predict_ensemble.py \
    ../data/lcl/merged_sequence_0.fna.gz \
    ../data/lcl/merged_orientation_index_logits_0.npz --gpu 1 --model_dir ensemble_models_logits/
```
