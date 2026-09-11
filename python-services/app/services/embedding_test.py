from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

texts = [
    "Python is a programming language.",
    "Python is used for software development.",
    "The weather is cold today.",
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=False,
)

similarities = cosine_similarity(embeddings)

print("\n========== EMBEDDING INFO ==========")

for text, embedding in zip(texts, embeddings):
    print("=" * 60)
    print("TEXT:", text)
    print("DIMENSION:", len(embedding))
    print("FIRST 5 VALUES:", embedding[:5])


print("\n========== COSINE SIMILARITY ==========")

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print(
            f"{i} <-> {j}: "
            f"{similarities[i][j]:.4f}"
        )