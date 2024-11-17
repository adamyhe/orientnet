# ORIENTNET

For predicting/attributing sequence contribution to promoter orientation.

## Benchmark

```bash
python predict_ensemble.py \
    ../data/lcl/merged_sequence_0.fna.gz \
    ../data/lcl/merged_orientation_index_logits_0.npz --gpu 1 --model_dir ensemble_models_logits/
python benchmark_orientnet.py
python benchmark_clipnet_baseline.py
```

## Attribution

DeepSHAP

```bash
python calculate_deepshap.py ../data/lcl/merged_sequence_0.fna.gz ../data/lcl/merged_sequence_0_orientnet_deepshap.npz ../data/lcl/merged_sequence_0_ohe.npz --model_dir ensemble_models_logits_v0/ --gpu 1
```

Modisco

```bash
cd ../data/lcl/
export NUMBA_NUM_THREADS=16
time modisco motifs \
    -s merged_sequence_0_ohe.npz \
    -a merged_sequence_0_orientnet_deepshap.npz \
    -n 1000000 -l 50 -v \
    -o merged_sequence_0_orientnet_modisco.h5
time modisco report \
    -i merged_sequence_0_orientnet_modisco.h5 \
    -o merged_sequence_0_orientnet_modisco/ \
    -m /home2/ayh8/data/JASPAR/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt
```
