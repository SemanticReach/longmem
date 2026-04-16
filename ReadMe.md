HyperBinder on LongMemEval

We evaluated HyperBinder on LongMemEval, a 500-question benchmark designed to measure long-term memory retrieval in conversational AI systems.

HyperBinder achieves 99–100% accuracy across runs, including a verified 500/500 (100%) with a fine-tuned configuration and 497/500 (99.4%) in our primary evaluation.

By comparison, prior systems typically range from 71% to 94% on the same benchmark.

This performance comes from HyperBinder’s dual-slot weighted semantic search, which jointly optimizes similarity over both the query and content fields. The result is near-perfect recall without requiring graph construction, multi-stage retrieval, or chain-of-thought prompting.

You can explore the full evaluation code here:
https://github.com/Jmp13033/longmem

Try it yourself:
Request an API key at questions@semantic-reach.io



