# src/dataset.py
from datasets import load_dataset as hf_load_dataset
from typing import List, Tuple, Optional
import random


def load_gsm8k(
    split: str = "train",
    config: str = "main",
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    GSM-8K Q/A + Chain-of-Thought 프롬프트
    """
    ds = hf_load_dataset("openai/gsm8k", config, split=split)
    if max_samples:
        ds = ds.select(range(max_samples))

    tmpl = "Q: {q}\nA: Let's think step by step."
    return [
        (tmpl.format(q=ex["question"].strip()),
         ex["answer"].split("####")[-1].strip().rstrip("."))
        for ex in ds
    ]


def load_cnn_dm(
    split: str = "train",
    version: str = "3.0.0",
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    CNN/DailyMail 2–4 문장 요약
    """
    ds = hf_load_dataset("cnn_dailymail", version, split=split)
    if max_samples:
        ds = ds.select(range(max_samples))

    tmpl = (
        "Summarize the following news article in 2–4 sentences.\n"
        "Article:\n{article}\n\nSummary:"
    )
    return [(tmpl.format(article=ex["article"].strip()), ex["highlights"].strip()) for ex in ds]


def load_wmt16(
    split: str = "train",
    config: str = "ro-en",
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    WMT-16(ro-en) 영어 → 루마니아어 번역
    """
    ds = hf_load_dataset("wmt16", config, split=split)
    if max_samples:
        ds = ds.select(range(max_samples))

    tmpl = (
        "Translate the following sentence from English to Romanian.\n"
        "English: {src}\nRomanian:"
    )
    return [
        (tmpl.format(src=ex["translation"]["en"].strip()),
         ex["translation"]["ro"].strip())
        for ex in ds
    ]

def load_human_eval(
    split: str = "test",
    repo_id: str = "openai/openai_humaneval",
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    ds = hf_load_dataset(repo_id, split=split)
    if max_samples:
        ds = ds.select(range(max_samples))

    # 그대로 반환 (별도 템플릿 불필요)
    return [
        (ex["prompt"].strip(), ex["canonical_solution"].strip())
        for ex in ds
    ]

def load_xsum(
    split: str = "train",
    version: str = "1.2.0",
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    XSum 한 문장 헤드라인 요약
    """
    ds = hf_load_dataset("xsum", version, split=split)
    if max_samples:
        ds = ds.select(range(max_samples))

    tmpl = (
        "Summarize the following news article in one concise sentence.\n"
        "Article:\n{doc}\n\nSummary:"
    )
    return [(tmpl.format(doc=ex["document"].strip()), ex["summary"].strip()) for ex in ds]

def _attach_fewshot(
    data: List[Tuple[str, str]],
    k: int,
    seed: int = 42,
    delimiter: str = "\n\n",
) -> List[Tuple[str, str]]:
    """
    같은 split 안에서 k개 예시를 골라 각 프롬프트 앞에 붙인다.
    (자기 자신은 제외)
    """
    if k <= 0 or len(data) <= 1:
        return data

    rng = random.Random(seed)
    new_data: List[Tuple[str, str]] = []
    indices = list(range(len(data)))

    for cur_idx, (prompt, answer) in enumerate(data):
        pool = indices.copy()
        pool.remove(cur_idx)                 # 자기 자신 제외
        if len(pool) < k:
            few_idx = pool                   # 데이터가 k보다 적으면 가능한 만큼
        else:
            few_idx = rng.sample(pool, k)

        fewshot_blocks = []
        for j, fi in enumerate(few_idx, 1):
            ex_prompt, ex_ans = data[fi]
            fewshot_blocks.append(
                f"### Example {j}\n{ex_prompt}\n{ex_ans}"
            )

        full_prompt = delimiter.join(fewshot_blocks + [prompt])
        new_data.append((full_prompt, answer))

    return new_data


def load_dataset(args) -> Tuple[List[Tuple[str, str]], int]:
    """
    args.dataset 에 맞는 로더 호출 + (선택) few-shot 예시 부착
    반환 → (data_list, max_length)
    """
    mapping = {
        "gsm8k": (load_gsm8k, 256),
        "cnn_dm": (load_cnn_dm, 128),
        "wmt16": (load_wmt16, 64),
        "xsum":  (load_xsum, 128),
        "human_eval": (load_human_eval, 512),
    }
    if args.dataset not in mapping:
        raise ValueError(f"Invalid dataset: {args.dataset!r}")

    loader_fn, default_max = mapping[args.dataset]
    max_len = default_max if args.max_length is None else args.max_length

    # ① 데이터 로드
    data = loader_fn(
        split=args.split,
        max_samples=None,
    )
    
    if args.max_samples is not None and args.max_samples < len(data):
        rng = random.Random(getattr(args, "seed", args.shuffle_seed))
        data = rng.sample(data, args.max_samples)

    n_fs   = getattr(args, "n_fewshot", 0)
    seed   = getattr(args, "seed", args.shuffle_seed)
    data   = _attach_fewshot(data, k=n_fs, seed=seed)

    return data, max_len
