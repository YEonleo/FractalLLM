<img width="1222" height="81" alt="제목 없음" src="https://github.com/user-attachments/assets/dc9c2f9f-64e9-43ad-bb1b-2ae3b90e9c40" />
# FractalLLM

**Lossless Self-Speculative Decoding with Layer-Embedded Self-Compression**

---


## Abstract
<img width="1306" alt="method" src="https://github.com/user-attachments/assets/1cd8bc57-c846-4ed7-99a3-fa00e0ea70e7" />

Autoregressive decoding in large language models (LLMs) requires a full forward pass per generated token, leading to high inference latency. **FractalLLM** injects lightweight “Fractal Layers”—compressed sub-models—at selected decoder depths to generate multiple draft tokens in parallel, then verifies them in one full-precision pass. This **lossless** speculative decoding matches the original outputs exactly while reducing the number of forward passes. Experiments on GSM8K, XSUM, CNN/DailyMail and HumanEval demonstrate up to **2.47×** speed-ups with negligible or even lower total FLOPs.

---

## Key Ideas

1. **Layer-Embedded Compression**  
   Inject small 8-bit Fractal Layers inside every 4th decoder layer (e.g. 7 layers in a 28-layer model).  
2. **Draft & Verify**  
   - **Draft**: generate _w_ tokens in parallel via the Fractal Layers.  
   - **Verify**: one full-precision pass of the original model corrects any mistakes—outputs remain bit-exact.  
3. **Trade-off Tuning**  
   - **# of Fractal Layers (n)** ≈ total_layers/4  
   - **Draft window (w)** ≈ _n_–1 (empirically 6–8 tokens)  
   - Typical verify iterations per sequence: ~30–40  

---

## Experimental Results

### Throughput & Speed-up on Summarization & Reasoning

| Model        | Task     | Baseline Tok/s | Fractal Tok/s | Speed-up |
|--------------|----------|---------------:|--------------:|---------:|
| LLaMA-3 3B   | GSM8K    |         12.20  |       14.60   |   1.20×  |
| LLaMA-3 3B   | XSUM     |          5.74  |        8.49   |   1.48×  |
| LLaMA-3 3B   | CNN/DM   |          3.13  |        5.51   |   1.69×  |
| LLaMA-3 8B   | GSM8K    |          4.26  |        7.47   |   1.75×  |
| LLaMA-3 8B   | XSUM     |          2.25  |        4.06   |   1.80×  |
| LLaMA-3 8B   | CNN/DM   |          1.01  |        2.15   |   2.12×  |

### Code-Generation (HumanEval)

| Model               | Baseline Tok/s | Fractal Tok/s | Speed-up |
|---------------------|---------------:|--------------:|---------:|
| CodeLLaMA-13B       |          1.79  |        4.43   |   2.47×  |

---

## Installation

```bash
git clone https://github.com/anonymous/fractal-llm.git
cd fractal-llm

# In your repo root (where fractal_env.yaml lives):
conda env create -f fractal_env.yaml
conda activate fractal_env

# Verify
conda list | grep fractal_env

