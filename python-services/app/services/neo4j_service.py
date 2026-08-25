"""
سرویس اتصال به Neo4j برای ذخیره گراف دانش
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Neo4jService:
    """سرویس مدیریت گراف در Neo4j"""
    
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="password"):
        """مقداردهی اولیه اتصال به Neo4j"""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self._connect()
    
    def _connect(self):
        """ایجاد اتصال به Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # تست اتصال
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            print("   ✅ Connected to Neo4j successfully")
        except Exception as e:
            print(f"   ❌ Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def save_document(self, doc_id: int, title: str, user_id: int, session_id: int):
        """ذخیره یک سند در Neo4j"""
        if not self.driver:
            print("   ⚠️ Neo4j not connected, skipping save")
            return False
        
        try:
            with self.driver.session() as session:
                session.run("""
                    CREATE (d:Document {
                        id: $doc_id,
                        title: $title,
                        user_id: $user_id,
                        session_id: $session_id,
                        created_at: datetime()
                    })
                """, {
                    'doc_id': doc_id,
                    'title': title,
                    'user_id': user_id,
                    'session_id': session_id
                })
                print(f"   ✅ Document {doc_id} saved to Neo4j")
                return True
        except Exception as e:
            print(f"   ❌ Error saving document: {e}")
            return False
    
    def save_entities(self, doc_id: int, entities: List[Dict[str, Any]]):
        """ذخیره موجودیت‌ها در Neo4j"""
        if not self.driver or not entities:
            return False
        
        try:
            with self.driver.session() as session:
                for entity in entities:
                    session.run("""
                        MATCH (d:Document {id: $doc_id})
                        CREATE (e:Entity {
                            name: $name,
                            type: $type,
                            description: $description
                        })
                        CREATE (d)-[:CONTAINS]->(e)
                    """, {
                        'doc_id': doc_id,
                        'name': entity.get('name', ''),
                        'type': entity.get('type', 'Concept'),
                        'description': entity.get('description', '')
                    })
                print(f"   ✅ Saved {len(entities)} entities to Neo4j")
                return True
        except Exception as e:
            print(f"   ❌ Error saving entities: {e}")
            return False
    
    def close(self):
        """بستن اتصال به Neo4j"""
        if self.driver:
            self.driver.close()