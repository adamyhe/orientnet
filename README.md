# ORIENTNET

For predicting/attributing sequence contribution to promoter orientation.

## Benchmark

```bash
python predict_ensemble.py \
    ../data/lcl/merged_sequence_0.fna.gz \
    ../data/lcl/merged_orientation_index_logits_0.npz \
    --model_dir ensemble_models_logits/ \
    --gpu 1

python predict_ensemble.py \
    ../data/lcl/merged_sequence_0.fna.gz \
    ../data/lcl/merged_orientation_index_logits_rescale_true_0.npz \
    --model_dir ensemble_models_logits_rescale_true/ \
    --gpu 1 

python predict_ensemble.py \
    ../data/lcl/merged_sequence_0.fna.gz \
    ../data/lcl/merged_orientation_index_logits_freeze_base_0.npz \
    --model_dir ensemble_models_logits_freeze_base/ \
    --gpu 1 

python benchmark_orientnet.py
python benchmark_clipnet_baseline.py
```

## Attribution

DeepSHAP

```bash
python calculate_deepshap.py ../data/lcl/merged_sequence_0.fna.gz ../data/lcl/merged_sequence_0_orientnet_deepshap.npz ../data/lcl/merged_sequence_0_ohe.npz --model_dir ensemble_models_logits_v0/ --gpu 1

python calculate_deepshap.py ../data/lcl/all_tss_windows_reference_seq.fna.gz ../data/lcl/all_tss_windows_deepshap.npz ../data/lcl/all_tss_windows_ohe.npz --model_dir ensemble_models_logits_rescale_true/ --gpu 1
```

Modisco

```bash
cd ../data/lcl/
export NUMBA_NUM_THREADS=32
time modisco motifs \
    -s all_tss_windows_ohe.npz \
    -a all_tss_windows_orientnet_deepshap.npz \
    -n 1000000 -l 50 -v \
    -o all_tss_windows_orientnet_deepshap.h5
time modisco report \
    -i all_tss_windows_orientnet_deepshap.h5 \
    -o all_tss_windows_orientnet_deepshap/ \
    -m /home2/ayh8/data/JASPAR/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt
```
