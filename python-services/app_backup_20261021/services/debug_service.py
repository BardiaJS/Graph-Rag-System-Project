"""
Debug service — چاپ گره‌ها، یال‌ها، و وکتورها توی ترمینال
"""
from app.services.embedding_service import client, COLLECTION_NAME
from app.services.graph_service import get_driver
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def print_chunks_summary(chunks: list):
    """خلاصه chunks رو چاپ کن"""
    print(f"\n{'='*70}")
    print(f"✂️  CHUNKS SUMMARY")
    print(f"{'='*70}")
    print(f"Total chunks: {len(chunks)}")

    headings_count = Counter()
    pages = set()
    total_chars = 0

    for c in chunks:
        for h in c.get('headings', []):
            headings_count[h] += 1
        if c.get('page'):
            pages.add(c['page'])
        total_chars += len(c.get('text', ''))

    print(f"Total chars: {total_chars:,}")
    print(f"Unique pages: {len(pages)}")
    print(f"Unique headings: {len(headings_count)}")

    print(f"\n📂 Top 10 headings:")
    for h, cnt in headings_count.most_common(10):
        print(f"  [{cnt:3d}] {h[:60]}")

    print(f"\n{'='*70}\n")


def print_qdrant_points(document_id: int = None, limit: int = 10):
    """همه points Qdrant رو چاپ کن"""
    print(f"\n{'='*70}")
    print(f"📊 QDRANT POINTS")
    print(f"{'='*70}")

    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total points: {info.points_count}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    query_filter = None
    if document_id:
        from qdrant_client import models
        query_filter = models.Filter(
            must=[models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )]
        )

    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        print(f"❌ Scroll error: {e}")
        return

    print(f"\nنمایش {len(points)} point:")
    for i, p in enumerate(points, 1):
        payload = p.payload or {}
        print(f"\n  [{i}] ID: {p.id}")
        print(f"      chunk_index: {payload.get('chunk_index')}")
        print(f"      page: {payload.get('page')}")
        print(f"      headings: {payload.get('headings', [])}")
        print(f"      text: {payload.get('text', '')[:80]}...")

    print(f"\n{'='*70}\n")


def print_neo4j_graph(document_id: int = None, limit: int = 10):
    """همه nodes و edges Neo4j رو چاپ کن"""
    print(f"\n{'='*70}")
    print(f"🕸️  NEO4J GRAPH")
    print(f"{'='*70}")

    driver = get_driver()
    with driver.session() as session:
        # Nodes
        if document_id:
            result = session.run("""
                MATCH (n)
                WHERE n.document_id = $doc_id OR n.id = $doc_id
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY label
            """, doc_id=document_id)
        else:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY label
            """)

        print("\n📦 Nodes:")
        for r in result:
            if r['label']:
                print(f"  {r['label']:15s} → {r['count']}")

        # Edges
        if document_id:
            result = session.run("""
                MATCH (a)-[r]->(b)
                WHERE a.document_id = $doc_id OR b.document_id = $doc_id
                RETURN type(r) AS type, count(r) AS count
                ORDER BY type
            """, doc_id=document_id)
        else:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY type
            """)

        print("\n🔗 Edges:")
        for r in result:
            if r['type']:
                print(f"  {r['type']:15s} → {r['count']}")

        # نمونه
        if document_id:
            result = session.run("""
                MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
                OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
                RETURN d.title AS doc, s.heading AS section, count(c) AS chunks
                LIMIT $limit
            """, doc_id=document_id, limit=limit)

            print(f"\n🌳 نمونه Sectionها (limit={limit}):")
            for r in result:
                print(f"  📄 {r['doc']}")
                print(f"    └─ 📂 {r['section']} ({r['chunks']} chunks)")

    print(f"\n{'='*70}\n")


def print_search_results(query: str, results: list):
    """نتایج search رو قشنگ چاپ کن"""
    print(f"\n{'='*70}")
    print(f"🔍 SEARCH: \"{query}\"")
    print(f"{'='*70}")
    print(f"Total results: {len(results)}")

    vector_results = [r for r in results if r.get('source') != 'graph']
    graph_results = [r for r in results if r.get('source') == 'graph']

    print(f"  📊 Vector results: {len(vector_results)}")
    print(f"  🕸️  Graph results:  {len(graph_results)}")

    print(f"\n📋 همه نتایج:")
    for i, r in enumerate(results, 1):
        source = r.get('source', 'vector')
        icon = '📊' if source == 'vector' else '🕸️'
        score = r.get('score', 0)
        page = r.get('page', '?')
        idx = r.get('chunk_index', '?')
        headings = r.get('headings', [])
        heading = headings[0] if headings else '-'
        text = r.get('text', '')[:60]

        print(f"\n  [{i}] {icon} score={score:.3f} | page={page} | idx={idx}")
        print(f"      heading: {heading}")
        print(f"      text: {text}...")

    print(f"\n{'='*70}\n")