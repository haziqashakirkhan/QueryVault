"""Step 3: turn text into vectors with a sentence-transformer model.

A sentence-transformer is a neural network trained so that sentences with
similar meaning produce vectors that point in similar directions. The model is
downloaded once and then runs locally: no API calls, no cost.
"""


class Embedder:
    def __init__(self, model_name: str):
        # Imported here so the app (and the tests) start fast when a fake
        # embedder is injected.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True gives every vector length 1, so cosine
        # similarity becomes a simple dot product.
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
