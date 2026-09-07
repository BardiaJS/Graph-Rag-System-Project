import hashlib
import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


logger = logging.getLogger(__name__)


class Neo4jService:
    """
    سرویس مدیریت Knowledge Graph در Neo4j.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
    ):

        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None

        self._connect()

    # ==========================================================
    # Connection
    # ==========================================================

    def _connect(self) -> None:

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(
                    self.username,
                    self.password,
                ),
            )

            self.driver.verify_connectivity()

            print(
                "   ✅ Connected to Neo4j successfully"
            )

        except Exception as e:

            print(
                f"   ❌ Failed to connect to Neo4j: {e}"
            )

            self.driver = None

    # ==========================================================
    # Entity ID
    # ==========================================================

    @staticmethod
    def make_entity_id(
        name: str,
        entity_type: str,
    ) -> str:

        raw = (
            f"{entity_type.lower().strip()}:"
            f"{name.lower().strip()}"
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    # ==========================================================
    # Save Document
    # ==========================================================

    def save_document(
        self,
        doc_id: int,
        title: str,
        user_id: int,
        session_id: int,
    ) -> bool:

        if not self.driver:
            print("   ⚠️ Neo4j not connected")
            return False

        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (d:Document {id: $doc_id})

                    ON CREATE SET
                        d.created_at = datetime()

                    SET
                        d.title = $title,
                        d.user_id = $user_id,
                        d.session_id = $session_id,
                        d.updated_at = datetime()
                    """,
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                )

            print(
                f"   ✅ Document {doc_id} saved to Neo4j"
            )
            return True

        except Exception:
            logger.exception(
                f"Error saving document {doc_id}"
            )
            return False

    # ==========================================================
    # Save Entities
    # ==========================================================

    def save_entities(
        self,
        doc_id: int,
        entities: List[Dict[str, Any]],
    ) -> bool:

        if not self.driver:
            return False

        if not entities:
            return True

        try:

            with self.driver.session() as session:

                for entity in entities:

                    name = str(
                        entity.get("name", "")
                    ).strip()

                    entity_type = str(
                        entity.get(
                            "type",
                            "concept",
                        )
                    ).strip()

                    description = str(
                        entity.get(
                            "description",
                            "",
                        )
                    ).strip()

                    if not name:
                        continue

                    entity_id = (
                        self.make_entity_id(
                            name,
                            entity_type,
                        )
                    )

                    session.run(
                        """
                        MATCH (d:Document {id: $doc_id})

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

                        MERGE (d)-[:CONTAINS]->(e)
                        """,
                        {
                            "doc_id": doc_id,
                            "entity_id": entity_id,
                            "name": name,
                            "type": entity_type,
                            "description": description,
                        },
                    )

            print(
                f"   ✅ Saved {len(entities)} entities "
                f"for document {doc_id}"
            )

            return True

        except Exception:
            logger.exception(
                f"Error saving entities for "
                f"document {doc_id}"
            )
            return False

    # ==========================================================
    # Search entities
    # ==========================================================

    def search_entities(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:

        if not self.driver:
            return []

        query = query.strip()

        if not query:
            return []

        # کلمات meaningful سؤال
        words = [
            word.lower()
            for word in query.split()
            if len(word.strip()) >= 3
        ]

        if not words:
            return []

        try:

            cypher = """
            MATCH (d:Document)-[:CONTAINS]->(e:Entity)

            WHERE
                (
                    $document_ids IS NULL
                    OR d.id IN $document_ids
                )
                AND any(
                    word IN $words
                    WHERE
                        toLower(e.name)
                        CONTAINS word
                )

            RETURN DISTINCT
                e.id AS entity_id,
                e.name AS entity_name,
                e.type AS entity_type,
                e.description AS description,
                d.id AS doc_id,
                d.title AS document_title

            LIMIT $limit
            """

            params = {
                "document_ids": (
                    document_ids
                    if document_ids
                    else None
                ),
                "words": words,
                "limit": limit,
            }

            results = []

            with self.driver.session() as session:

                records = session.run(
                    cypher,
                    params,
                )

                for record in records:

                    results.append(
                        {
                            "entity_id": record[
                                "entity_id"
                            ],
                            "entity_name": record[
                                "entity_name"
                            ],
                            "entity_type": record[
                                "entity_type"
                            ],
                            "description": record[
                                "description"
                            ],
                            "doc_id": record[
                                "doc_id"
                            ],
                            "document_title": record[
                                "document_title"
                            ],
                            "score": 1.0,
                            "type": "graph",
                            "text": (
                                f"{record['entity_name']} "
                                f"({record['entity_type']})"
                            ),
                        }
                    )

            return results

        except Exception:
            logger.exception(
                "Neo4j entity search failed"
            )
            return []

    # ==========================================================
    # Delete document
    # ==========================================================

    def delete_document(
        self,
        doc_id: int,
    ) -> bool:

        if not self.driver:
            return False

        try:

            with self.driver.session() as session:

                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    DETACH DELETE d
                    """,
                    {
                        "doc_id": doc_id
                    },
                )

            return True

        except Exception:
            logger.exception(
                "Neo4j document deletion failed"
            )
            return False

    # ==========================================================
    # Close
    # ==========================================================

    def close(self) -> None:

        if self.driver:

            self.driver.close()
            self.driver = None