# ORIENTNET

For predicting/attributing sequence contribution to promoter orientation.

## Attribution

```bash
python calculate_deepshap.py \
    ../data/lcl/merged_sequence_0.fna.gz \
    ../data/lcl/merged_orientation_deepshap_0.npz \
    ../data/lcl/merged_sequence_ohe_0.npz \
    --model_dir ensemble_models_v0/ \
    --gpu 1
```
