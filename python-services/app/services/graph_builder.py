from typing import List, Dict, Any
import hashlib
import json

class GraphBuilder:
    """ساخت گراف دانش از اسناد استخراج شده"""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    
    def build_graph_from_document(self, doc_data: Dict[str, Any], doc_id: int):
        """ساخت گراف از یک سند"""
        
        with self.driver.session() as session:
            # ایجاد گره سند
            session.run("""
                CREATE (d:Document {
                    id: $doc_id,
                    title: $title,
                    created_at: datetime()
                })
            """, {
                'doc_id': doc_id,
                'title': doc_data.get('title', f'Document_{doc_id}')
            })
            
            # استخراج موجودیت‌ها
            entities = self._extract_entities(doc_data['pages'])
            
            # ایجاد گره‌های موجودیت و روابط
            for entity in entities:
                session.run("""
                    MATCH (d:Document {id: $doc_id})
                    CREATE (e:Entity {
                        id: $entity_id,
                        name: $name,
                        type: $type,
                        description: $description
                    })
                    CREATE (d)-[:CONTAINS]->(e)
                """, {
                    'doc_id': doc_id,
                    'entity_id': f"entity_{hashlib.md5(entity['name'].encode()).hexdigest()}",
                    'name': entity['name'],
                    'type': entity.get('type', 'Concept'),
                    'description': entity.get('description', '')
                })
    
    def _extract_entities(self, pages: List[Dict]) -> List[Dict]:
        """استخراج موجودیت‌ها از متن (با الگوهای ساده)"""
        entities = []
        
        # برای شروع، از الگوهای ساده استفاده می‌کنیم
        # بعداً می‌توانیم از NER استفاده کنیم
        patterns = {
            'method': r'\b(method|approach|technique|algorithm)\b',
            'dataset': r'\b(dataset|data|benchmark)\b',
            'metric': r'\b(accuracy|precision|recall|f1|score)\b'
        }
        
        for page in pages:
            for col in page.get('content', []):
                text = col.get('text', '')
                
                for entity_type, pattern in patterns.items():
                    import re
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities.append({
                            'name': match,
                            'type': entity_type,
                            'description': ''
                        })
        
        # حذف تکراری‌ها
        unique_entities = []
        seen = set()
        for entity in entities:
            key = f"{entity['name']}_{entity['type']}"
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities