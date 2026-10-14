# Apertus on the DGX Spark

1. Copy this folder to the Spark (`scp -r spark user@spark:~/apertus`), then on the Spark:
   ```bash
   cd ~/apertus && cp .env.example .env && nano .env        # HF_TOKEN
   docker compose pull                                       # image tag latest-arm64
   docker compose up -d && docker compose logs -f            # wait for "Application startup complete"
   ```
2. Test from your PC: `http://<spark-ip>:8000/v1/models`, then set in the coach's `.env`:
   `LLM_BASE_URL_DOCKER=http://<spark-ip>:8000/v1` (the test scripts use the same URL).

Memory is shared: `VLLM_GPU_UTIL=0.5` ≈ 60 GB for vLLM (bf16 weights ~18 GB + KV cache),
the rest stays free for an evaluation judge model and LoRA experiments.
