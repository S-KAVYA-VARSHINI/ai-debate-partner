
import faiss
from sentence_transformers import SentenceTransformer


class DebateRAG:

    def __init__(self, knowledge_file):
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        with open(knowledge_file, "r", encoding="utf-8") as f:
            knowledge_text = f.read()

        self.chunks = self.create_chunks(knowledge_text)

        embeddings = self.embedding_model.encode(
            self.chunks,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def create_chunks(self, text):
        sections = text.split("DEBATE TOPIC:")

        chunks = []

        for section in sections:
            section = section.strip()

            if section:
                chunks.append("DEBATE TOPIC:" + section)

        return chunks

    def retrieve(self, query, k=2):
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        return [
            self.chunks[idx]
            for idx in indices[0]
        ]
