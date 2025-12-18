CUDA_VISIBLE_DEVICES=3 python -m debugpy --listen 5678 src/generate.py \
    --vocab results/tiny_stories_10k_vocab/vocab.pkl \
    --merges results/tiny_stories_10k_vocab/merges.pkl \
    --sampling --max_new_tokens 256 --sampling_topp 0.95 --temperature 0.6 \
    --input_text "Once upon a time," \
    --model_ckpt "results/training/TinyStories/TS_lr_5e-4_1e-5_warmup3K_anneal_30K_vocab_10K/checkpoint_1.443.pt" \
    --vocab_size 10000 --nheads 16 --context_length 256 --use_cuda \
    --rope_theta 10000 --d_model 512 --d_ff 1344 --nlayers 4