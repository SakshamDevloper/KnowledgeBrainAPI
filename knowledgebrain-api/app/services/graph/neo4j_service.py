"""
KnowledgeBrain — Neo4j Knowledge Graph Service
CRUD operations and queries for the industrial knowledge graph.
"""

import asyncio
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("neo4j_service")


class Neo4jService:
    """Manages Neo4j graph database operations."""

    def __init__(self, driver: Optional[AsyncDriver] = None):
        self.driver = driver
        self._owned_driver = False

    async def connect(self):
        """Initialize the Neo4j async driver."""
        if self.driver is None:
            settings = get_settings()
            self.driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            self._owned_driver = True
            logger.info(f"Connected to Neo4j at {settings.neo4j_uri}")

    async def close(self):
        """Close the Neo4j driver."""
        if self.driver and self._owned_driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def ensure_indexes(self):
        """Create indexes and constraints for performance."""
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Equipment) ON (e.tag_id)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.doc_id)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Regulation) ON (r.code)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Incident) ON (i.incident_id)",
            "CREATE INDEX IF NOT EXISTS FOR (w:WorkOrder) ON (w.wo_id)",
        ]
        async with self.driver.session() as session:
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
        logger.info("Neo4j indexes ensured")

    # ── Node CRUD ──────────────────────────────────

    async def upsert_equipment(self, tag_id: str, **properties) -> dict:
        """
        Create or update an Equipment node.
        Cypher: MERGE (e:Equipment {tag_id: $tag_id}) SET e += $props RETURN e
        """
        query = """
        MERGE (e:Equipment {tag_id: $tag_id})
        SET e += $props, e.updated_at = datetime()
        RETURN e {.*} AS equipment
        """
        props = {"tag_id": tag_id, **properties}
        async with self.driver.session() as session:
            result = await session.run(query, tag_id=tag_id, props=props)
            record = await result.single()
            return dict(record["equipment"]) if record else {}

    async def upsert_document(self, doc_id: str, **properties) -> dict:
        """Create or update a Document node."""
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d += $props, d.updated_at = datetime()
        RETURN d {.*} AS document
        """
        props = {"doc_id": doc_id, **properties}
        async with self.driver.session() as session:
            result = await session.run(query, doc_id=doc_id, props=props)
            record = await result.single()
            return dict(record["document"]) if record else {}

    async def upsert_regulation(self, code: str, **properties) -> dict:
        """Create or update a Regulation node."""
        query = """
        MERGE (r:Regulation {code: $code})
        SET r += $props, r.updated_at = datetime()
        RETURN r {.*} AS regulation
        """
        props = {"code": code, **properties}
        async with self.driver.session() as session:
            result = await session.run(query, code=code, props=props)
            record = await result.single()
            return dict(record["regulation"]) if record else {}

    async def upsert_person(self, name: str, **properties) -> dict:
        """Create or update a Person node."""
        query = """
        MERGE (p:Person {name: $name})
        SET p += $props, p.updated_at = datetime()
        RETURN p {.*} AS person
        """
        props = {"name": name, **properties}
        async with self.driver.session() as session:
            result = await session.run(query, name=name, props=props)
            record = await result.single()
            return dict(record["person"]) if record else {}

    async def upsert_incident(self, incident_id: str, **properties) -> dict:
        """Create or update an Incident node."""
        query = """
        MERGE (i:Incident {incident_id: $incident_id})
        SET i += $props, i.updated_at = datetime()
        RETURN i {.*} AS incident
        """
        props = {"incident_id": incident_id, **properties}
        async with self.driver.session() as session:
            result = await session.run(query, incident_id=incident_id, props=props)
            record = await result.single()
            return dict(record["incident"]) if record else {}

    async def upsert_work_order(self, wo_id: str, **properties) -> dict:
        """Create or update a WorkOrder node."""
        query = """
        MERGE (w:WorkOrder {wo_id: $wo_id})
        SET w += $props, w.updated_at = datetime()
        RETURN w {.*} AS work_order
        """
        props = {"wo_id": wo_id, **properties}
        async with self.driver.session() as session:
            result = await session.run(query, wo_id=wo_id, props=props)
            record = await result.single()
            return dict(record["work_order"]) if record else {}

    # ── Relationship CRUD ──────────────────────────

    async def create_relationship(
        self,
        from_label: str, from_key: str, from_value: str,
        to_label: str, to_key: str, to_value: str,
        rel_type: str,
        properties: Optional[dict] = None,
    ):
        """
        Create a relationship between two nodes.
        Example: create_relationship("Equipment", "tag_id", "P-101A",
                                      "Document", "doc_id", "doc-123",
                                      "HAS_DOCUMENT")
        """
        props = properties or {}
        query = f"""
        MATCH (a:{from_label} {{{from_key}: $from_value}})
        MATCH (b:{to_label} {{{to_key}: $to_value}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        RETURN type(r) AS rel_type
        """
        async with self.driver.session() as session:
            await session.run(
                query,
                from_value=from_value,
                to_value=to_value,
                props=props,
            )

    # ── Query Functions ────────────────────────────

    async def get_equipment_context(self, tag_id: str) -> dict:
        """
        Get all documents, work orders, incidents for an equipment tag.
        Returns a comprehensive context object for the copilot.
        """
        query = """
        MATCH (e:Equipment {tag_id: $tag_id})
        OPTIONAL MATCH (e)-[:HAS_DOCUMENT]->(d:Document)
        OPTIONAL MATCH (w:WorkOrder)-[:PERFORMED_ON]->(e)
        OPTIONAL MATCH (i:Incident)-[:INVOLVES]->(e)
        OPTIONAL MATCH (e)-[:SUBJECT_TO]->(r:Regulation)
        RETURN e {.*} AS equipment,
               collect(DISTINCT d {.*}) AS documents,
               collect(DISTINCT w {.*}) AS work_orders,
               collect(DISTINCT i {.*}) AS incidents,
               collect(DISTINCT r {.*}) AS regulations
        """
        async with self.driver.session() as session:
            result = await session.run(query, tag_id=tag_id)
            record = await result.single()
            if record:
                return {
                    "equipment": dict(record["equipment"]),
                    "documents": [dict(d) for d in record["documents"] if d],
                    "work_orders": [dict(w) for w in record["work_orders"] if w],
                    "incidents": [dict(i) for i in record["incidents"] if i],
                    "regulations": [dict(r) for r in record["regulations"] if r],
                }
            return {"equipment": None, "documents": [], "work_orders": [], "incidents": [], "regulations": []}

    async def find_similar_incidents(self, description: str, limit: int = 3) -> list[dict]:
        """
        Find incidents similar to a given description.
        Uses text containment as a basic similarity proxy.
        For production, this would use embedding similarity.
        """
        # Extract key terms (simple approach)
        terms = [w for w in description.lower().split() if len(w) > 3][:5]
        conditions = " OR ".join([f"toLower(i.description) CONTAINS '{t}'" for t in terms])

        if not conditions:
            return []

        query = f"""
        MATCH (i:Incident)
        WHERE {conditions}
        RETURN i {{.*}} AS incident
        LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(query, limit=limit)
            records = await result.data()
            return [dict(r["incident"]) for r in records]

    async def get_compliance_status(self, equipment_tag: str) -> list[dict]:
        """Get all applicable regulations and last inspection dates for equipment."""
        query = """
        MATCH (e:Equipment {tag_id: $tag_id})-[:SUBJECT_TO]->(r:Regulation)
        OPTIONAL MATCH (w:WorkOrder)-[:PERFORMED_ON]->(e)
        WHERE w.type = 'inspection'
        RETURN r {.*} AS regulation,
               max(w.date) AS last_inspection
        ORDER BY r.code
        """
        async with self.driver.session() as session:
            result = await session.run(query, tag_id=equipment_tag)
            records = await result.data()
            return [
                {
                    "regulation": dict(r["regulation"]),
                    "last_inspection": r.get("last_inspection"),
                }
                for r in records
            ]

    async def get_entity_neighborhood(self, node_label: str, node_key: str, node_value: str, depth: int = 2) -> dict:
        """Get the graph neighborhood around a node for visualization."""
        query = f"""
        MATCH path = (n:{node_label} {{{node_key}: $value}})-[*1..{depth}]-(m)
        RETURN nodes(path) AS nodes, relationships(path) AS relationships
        LIMIT 100
        """
        async with self.driver.session() as session:
            result = await session.run(query, value=node_value)
            records = await result.data()

            all_nodes = {}
            all_rels = []

            for record in records:
                for node in record.get("nodes", []):
                    node_dict = dict(node)
                    node_id = node_dict.get("tag_id") or node_dict.get("doc_id") or node_dict.get("name") or str(id(node))
                    all_nodes[node_id] = node_dict
                for rel in record.get("relationships", []):
                    all_rels.append({
                        "type": rel.type if hasattr(rel, 'type') else str(rel),
                    })

            return {
                "nodes": list(all_nodes.values()),
                "relationships": all_rels,
            }

    async def get_graph_stats(self) -> dict:
        """Get summary statistics of the knowledge graph."""
        query = """
        CALL {
            MATCH (e:Equipment) RETURN 'Equipment' AS label, count(e) AS count
            UNION ALL
            MATCH (d:Document) RETURN 'Document' AS label, count(d) AS count
            UNION ALL
            MATCH (r:Regulation) RETURN 'Regulation' AS label, count(r) AS count
            UNION ALL
            MATCH (p:Person) RETURN 'Person' AS label, count(p) AS count
            UNION ALL
            MATCH (i:Incident) RETURN 'Incident' AS label, count(i) AS count
            UNION ALL
            MATCH (w:WorkOrder) RETURN 'WorkOrder' AS label, count(w) AS count
        }
        RETURN label, count
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(query)
                records = await result.data()
                return {r["label"]: r["count"] for r in records}
        except Exception as e:
            logger.warning(f"Failed to get graph stats: {e}")
            return {}
