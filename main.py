import logging
import time
from collections import deque

import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from src.args import get_args
from src.dataset import load_dataset
from src.model import load_model_with_fractal, set_verify_mode, load_quantized_model
from src.generate import ParallelSPGenerator
from src.utils import FlopsCounter


def prepare_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    draft_tokens = None
    if args.decode_method == 'fractal':
        if args.draft_token == "[DRAFT]":
            draft_tokens = ["[DRAFT]"] * args.draft_len
            tokenizer.add_special_tokens({"additional_special_tokens": draft_tokens})
        elif args.draft_token == "[DRAFT{i}]":
            draft_tokens = [f"[DRAFT{i}]" for i in range(1, args.draft_len + 1)]
            tokenizer.add_special_tokens({"additional_special_tokens": draft_tokens})
        elif args.draft_token != "unk":
            draft_tokens = [args.draft_token] * args.draft_len

    return tokenizer, draft_tokens


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)

    ts = time.strftime("%y_%m-%d_%H:%M", time.localtime())
    if args.decode_method == "fractal":
        num_draft_layers = len(args.draft_layer_indexes)
        exp_name = (
            f"{ts}-{args.model_name}-{args.decode_method}-{args.dataset}-max_samples:{args.max_samples}-"
            f"{args.decomp_method}-draft_len:{args.draft_len}-num_draft_layers:{num_draft_layers}"
        )
    elif args.decode_method == "draft":
        exp_name = (
            f"{ts}-{args.model_name}-{args.decode_method}-{args.dataset}-max_samples:{args.max_samples}-"
            f"draft_len:{args.draft_len}"
        )
    else:
        exp_name = f"{ts}-{args.model_name}-{args.decode_method}-{args.dataset}-max_samples:{args.max_samples}"
    # 데이터 및 토크나이저 준비
    data_iter, max_length = load_dataset(args)
    data_list = list(data_iter)
    tokenizer, draft_tokens = prepare_tokenizer(args)

    # 모델 로드
    if args.decode_method == "fractal":
        model = load_model_with_fractal(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            tokenizer=tokenizer,
            device_map=args.device_map,
            experiment_config=args,
        )
        draft_mode_func = set_verify_mode
    elif args.decode_method == 'baseline':
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
        )
        draft_mode_func = None

    perf_base = {"times": [], "flops": [], "tokens": []}
    perf_spec = {"times": [], "flops": [], "tokens": []}

    if args.decode_method == "baseline":
        model.eval()
        flops_counter = FlopsCounter(model)
        
        
        for idx, (question, _) in enumerate(tqdm(data_list, desc="baseline")):
            flops_counter.reset()

            enc = tokenizer(question, return_tensors="pt").to(model.device)
            generated = enc["input_ids"].clone()
            prompt_len = generated.size(1)
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("토크나이저에 EOS 토큰이 지정되어 있지 않습니다.")

            t0 = time.time()
            with torch.no_grad():
                for _ in range(max_length):
                    logits = model(generated, use_cache=args.use_cache).logits[:, -1, :]
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                    ### EOS
                    if next_id.item() == tokenizer.eos_token_id:
                        break
                    generated = torch.cat([generated, next_id], dim=-1)

            elapsed = time.time() - t0

            flops = flops_counter.get_total_flops()
            new_tokens = generated.size(1) - prompt_len

            perf_base["times"].append(elapsed)
            perf_base["flops"].append(flops)
            perf_base["tokens"].append(new_tokens)
            
    elif args.decode_method == "fractal": 
        model.eval()
        flops_counter = FlopsCounter(model)
        
        for data in tqdm(data_list, desc="fractal"):
            
            performance_dict = {
                "model_forward_count": {"draft": 0, "verify": 0},
                "new_tokens": 0,
                "draft_time": 0.0,
                "verify_time": 0.0,
                # ==================== NEW ====================
                "total_accept_count": 0,    # 누적 맞은 토큰 수
                "total_checked_count": 0,   # 누적 검증한 토큰 수
                "accept_ratio": 0.0,        # 실시간 비율
                # ============================================
            }
            
            gen = ParallelSPGenerator(
                model=model,
                tokenizer=tokenizer,
                draft_mode_func=draft_mode_func,
                data_queue=deque([data]),
                args=args,
                draft_tokens=draft_tokens,
                performance_dict=performance_dict,
                max_length=max_length,
            )

            flops_counter.reset()
            t0 = time.time()
            with torch.no_grad():
                performance_dict = gen.forward()
            elapsed = time.time() - t0
            flops = flops_counter.get_total_flops()
            new_tokens = performance_dict["new_tokens"]

            perf_spec["times"].append(elapsed)
            perf_spec["flops"].append(flops)
            perf_spec["tokens"].append(new_tokens)

            log_data = {
                "mode":       "speculative",
                "elapsed":    elapsed,
                "flops":      flops,
                "new_tokens": new_tokens,
                **performance_dict,
            }


if __name__ == "__main__":
    main()
