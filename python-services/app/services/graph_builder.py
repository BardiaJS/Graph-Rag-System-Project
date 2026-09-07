from typing import Any, Dict, List

from app.services.neo4j_service import Neo4jService


class GraphBuilder:
    """
    ساخت Knowledge Graph در Neo4j.

    Entity ID generation باید با Neo4jService
    یکسان باشد.
    """

    def __init__(
        self,
        neo4j_driver,
    ):
        self.driver = neo4j_driver

    def build_graph_from_document(
        self,
        doc_data: Dict[str, Any],
        doc_id: int,
    ) -> Dict[str, Any]:

        title = doc_data.get(
            "title",
            f"Document_{doc_id}",
        )

        text = doc_data.get(
            "full_text",
            "",
        )

        entities = self._extract_entities(
            text
        )

        with self.driver.session() as session:

            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET
                    d.title = $title,
                    d.updated_at = datetime()
                ON CREATE SET
                    d.created_at = datetime()
                """,
                {
                    "doc_id": doc_id,
                    "title": title,
                },
            )

            for entity in entities:

                name = entity["name"]
                entity_type = entity["type"]

                entity_id = (
                    Neo4jService.make_entity_id(
                        name,
                        entity_type,
                    )
                )

                session.run(
                    """
                    MERGE (
                        e:Entity {id: $entity_id}
                    )

                    SET
                        e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.updated_at = datetime()

                    ON CREATE SET
                        e.created_at = datetime()

                    WITH e

                    MATCH (
                        d:Document {id: $doc_id}
                    )

                    MERGE (
                        d
                    )-[:CONTAINS]->(e)
                    """,
                    {
                        "doc_id": doc_id,
                        "entity_id": entity_id,
                        "name": name,
                        "type": entity_type,
                        "description": entity.get(
                            "description",
                            "",
                        ),
                    },
                )

        return {
            "entities": len(entities)
        }

    # ==========================================================
    # Entity extraction
    # ==========================================================

    def _extract_entities(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:

        if not text:
            return []

        patterns = {
            "technology": [
                "Python",
                "TensorFlow",
                "PyTorch",
                "Keras",
                "scikit-learn",
                "Transformers",
                "GPT",
                "BERT",
                "LLM",
                "NLP",
                "CNN",
                "RNN",
                "LSTM",
                "GAN",
                "VAE",
                "MLP",
                "SVM",
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ],

            "organization": [
                "Google",
                "Microsoft",
                "Amazon",
                "OpenAI",
                "DeepMind",
                "Meta",
                "IBM",
                "Intel",
                "NVIDIA",
                "AMD",
                "Apple",
                "Stanford",
                "MIT",
                "Berkeley",
                "Oxford",
                "Cambridge",
            ],

            "concept": [
                "machine learning",
                "deep learning",
                "artificial intelligence",
                "neural network",
                "computer vision",
                "natural language processing",
                "reinforcement learning",
                "transfer learning",
                "federated learning",
            ],

            "metric": [
                "accuracy",
                "precision",
                "recall",
                "F1-score",
                "AUC",
                "ROC",
                "perplexity",
                "BLEU",
                "ROUGE",
                "MSE",
                "MAE",
                "RMSE",
            ],

            "dataset": [
                "dataset",
                "benchmark",
                "corpus",
                "SQuAD",
                "ImageNet",
                "COCO",
                "MNIST",
            ],
        }

        entities = []
        seen = set()

        for entity_type, names in patterns.items():

            for name in names:

                pattern = (
                    r"(?<!\w)"
                    + __import__("re").escape(name)
                    + r"(?!\w)"
                )

                if __import__("re").search(
                    pattern,
                    text,
                    __import__("re").IGNORECASE,
                ):

                    key = (
                        entity_type,
                        name.lower(),
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    entities.append(
                        {
                            "name": name,
                            "type": entity_type,
                            "description": (
                                f"{name} extracted "
                                f"from document"
                            ),
                        }
                    )

        return entities