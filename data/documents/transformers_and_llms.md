# Transformers and Large Language Models

The transformer architecture was introduced in 2017 in the paper "Attention Is All You Need". Its key idea is self-attention, which lets every token in a sequence weigh how relevant every other token is. This allows the model to capture long-range relationships and to be trained efficiently in parallel.

## How an LLM generates text

A large language model, or LLM, works with tokens, which are pieces of words. Given the tokens so far, it predicts a probability for every possible next token, picks one, appends it and repeats. The context window is the maximum number of tokens the model can read at once, including both the prompt and the answer. Anything outside the window is invisible to the model.

## Encoders and decoders

Encoder models such as BERT read a whole text at once and produce a representation of it, which makes them good at understanding tasks. Decoder models such as GPT generate text one token at a time. Sentence-transformers are encoder models fine-tuned so that the output vector for a whole sentence captures its meaning, which is why they are used to create embeddings for search.

## Limits of LLMs

LLMs can hallucinate, meaning they produce fluent text that is not true. Their knowledge stops at a training cutoff date. Two common remedies are fine-tuning, which changes the model's weights using new data, and retrieval-augmented generation, which leaves the model unchanged and supplies fresh facts in the prompt. RAG is usually cheaper and easier to keep up to date.
