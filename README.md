# Melody1437-27B-NVFP4-MTP

NVFP4 / FP8 mixed-precision build of [ReadyArt/Melody1437-27B-v0.3](https://huggingface.co/ReadyArt/Melody1437-27B-v0.3),
text-only, 21.6 GB across 5 shards.

Weights: <https://huggingface.co/FangZhang5959/Melody1437-27B-NVFP4-MTP-text-only>

## What this is

The base model is a Qwen3.6-27B fine-tune. It is quantized with llm-compressor in two
groups: 8-bit for attention projections, `lm_head` and the last 8 MLP layers, 4-bit NVFP4
(group size 16) for the remaining MLP layers. The vision tower is stripped, which is why
the architecture is `Qwen3_5ForCausalLM` rather than the multimodal original.

One thing differs from a plain quantization: the upstream checkpoint declares an MTP head
in `config.json` but keeps its tensors mixed into the main shards, where the quantization
pass drops them and speculative decoding silently stops working. This build keeps the head
in a separate `model_mtp.safetensors` file that the index still references.

## How to run

Needs 2 × 22 GB GPUs, or a single 24 GB card at low concurrency. Measured on 2 × RTX 2080 Ti:
224K context with 2 concurrent sequences, roughly 37 tokens/s single stream.

The build requires a vLLM fork that emulates NVFP4 on Turing (SM 7.5); stock vLLM rejects the
quantization format outright. On a Blackwell card use the upstream bf16 weights instead.
Configure the invocation according to that branch's NVFP4 documentation.

Sampling values are documented in the model card on Hugging Face.

## Sampling

`temperature 1.0`, `top_p 0.95`, `top_k 20`, as shipped in `generation_config.json`.

With MTP enabled, vLLM rejects `min_p` and `logit_bias` with a 400. Everything else in the
sampling stack works.

## Limitations

- Text only. The vision tower is removed, image inputs are ignored.
- Personal, non-commercial use — inherited from the upstream license, see below.
- Adult content: the fine-tune is an adult roleplay model.

## License and credits

Apache-2.0, see [LICENSE](./LICENSE). The upstream repository adds its own terms, reproduced
verbatim in `LICENSE.txt` there: personal use only, no commercial use, 18+.

- Fine-tune: [ReadyArt/Melody1437-27B-v0.3](https://huggingface.co/ReadyArt/Melody1437-27B-v0.3)
- Base model: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
