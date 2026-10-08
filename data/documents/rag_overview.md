# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation, usually shortened to RAG, combines information retrieval with text generation. Instead of relying only on what a language model memorised during training, a RAG system first retrieves relevant passages from an external knowledge source and then gives those passages to the model as context for its answer.

## Why RAG is used

Language models have a knowledge cutoff and sometimes invent facts, which is called hallucination. RAG reduces hallucination because the answer is grounded in retrieved text. Knowledge can be updated by simply adding or replacing documents, with no retraining. RAG can also show its sources, so a reader can check where an answer came from.

## The four stages of a RAG pipeline

1. Ingestion: documents are loaded, split into chunks, converted into embeddings and stored in a vector database.
2. Retrieval: the user's question is embedded and the most similar chunks, the top-k results, are fetched from the vector database.
3. Augmentation: the retrieved chunks are inserted into the prompt together with the question.
4. Generation: the language model writes an answer using only the supplied context.

## Common failure modes

Poor chunking can split an answer across two chunks so neither chunk is useful alone. A top-k value that is too small can miss the right passage, while one that is too large adds noise. If the retrieved chunks are irrelevant, the model may still produce a confident but wrong answer, so prompts should tell the model to say when the context does not contain the answer.
