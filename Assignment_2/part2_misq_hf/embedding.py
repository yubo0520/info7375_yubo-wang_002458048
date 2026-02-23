from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._cache = {}

    def encode(self, text):
        if text in self._cache:
            return self._cache[text]
        emb = self.model.encode(text, normalize_embeddings=True)
        # print(f"  [emb] dim={emb.shape[0]}, text='{text[:50]}'")
        self._cache[text] = emb
        return emb


def cosine_sim(a, b):
    return float(np.dot(a, b))  # embeddings are L2-normed


def assign_cluster(embedding, clusters, threshold=0.9):
    # Eq.10-11
    if not clusters:
        return 0

    best_id = None
    best_sim = -1
    for cid, embs in clusters.items():
        if len(embs) == 1:
            centroid = embs[0]
        else:
            # use medoid instead of centroid
            emb_arr = np.array(embs)
            sim_matrix = emb_arr @ emb_arr.T
            avg_sims = sim_matrix.sum(axis=1)
            medoid_idx = int(np.argmax(avg_sims))
            centroid = embs[medoid_idx]

        sim = cosine_sim(embedding, centroid)
        if sim > best_sim:
            best_sim = sim
            best_id = cid

    if best_sim >= threshold:
        return best_id
    return max(clusters.keys()) + 1


if __name__ == "__main__":
    emb = Embedder()
    e1 = emb.encode("patient has fever and cough for 3 days")
    e2 = emb.encode("patient has runny nose and sneezing")
    e3 = emb.encode("patient has fever and difficulty breathing")
    print(f"dim: {e1.shape}")
    print(f"fever+cough vs runny nose: {cosine_sim(e1, e2):.3f}")
    print(f"fever+cough vs fever+breathing: {cosine_sim(e1, e3):.3f}")

    # test 
    clusters = {0: [e1]}
    cid = assign_cluster(e3, clusters, threshold=0.8)
    print(f"e3 -> cluster {cid} (0=same, 1=new)")
    cid2 = assign_cluster(e2, clusters, threshold=0.95)
    print(f"e2 -> cluster {cid2}")
