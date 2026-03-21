# HyperBinder Achieves Near-Perfect Score on LongMemEval

We evaluated HyperBinder on LongMemEval, a benchmark of 500 questions designed to test long-term memory retrieval in conversational AI systems. HyperBinder achieves **99%–100% accuracy across runs**, reaching a confirmed **500/500 (100%)** with fined tuned configuration and **497/500 (99.4%)** in our primary evaluation.

Prior systems have ranged from 71% to 94% on the same benchmark. HyperBinder's dual-slot weighted semantic search — blending similarity across both the question and content chunk fields — enables near-perfect recall without graph construction, multi-stage re-reading, or chain-of-thought prompting.

The full evaluation code is available at [github.com/Jmp13033/longmem](https://github.com/Jmp13033/longmem).