"""
KnowledgeBrain — Knowledge Graph Builder
Transforms extracted entities into Neo4j graph nodes and relationships.
"""

from typing import Optional

from app.models.document import ExtractedEntity, DocType
from app.services.graph.neo4j_service import Neo4jService
from app.utils.logging import get_logger

logger = get_logger("graph_builder")


class GraphBuilder:
    """Builds the knowledge graph from extracted document entities."""

    def __init__(self, neo4j_service: Neo4jService):
        self.neo4j = neo4j_service

    async def build_from_document(
        self,
        doc_id: str,
        filename: str,
        doc_type: DocType,
        entities: list[ExtractedEntity],
    ):
        """
        Build graph nodes and relationships from a document's extracted entities.

        Creates:
        - Document node
        - Equipment nodes + (Equipment)-[:HAS_DOCUMENT]->(Document)
        - Regulation nodes + (Document)-[:REFERENCES]->(Regulation)
        - Person nodes
        - WorkOrder nodes (if document_id entities match WO patterns)
        """
        # 1. Create Document node
        await self.neo4j.upsert_document(
            doc_id=doc_id,
            filename=filename,
            doc_type=doc_type.value,
        )

        # 2. Process entities by type
        equipment_tags = set()
        regulatory_refs = set()
        persons = set()
        work_orders = set()

        for entity in entities:
            if entity.entity_type == "equipment_tag":
                equipment_tags.add(entity.value)
            elif entity.entity_type == "regulatory_ref":
                regulatory_refs.add(entity.value)
            elif entity.entity_type == "person":
                persons.add(entity.value)
            elif entity.entity_type == "document_id" and entity.value.upper().startswith(("WO", "MWO", "PM")):
                work_orders.add(entity.value)

        # 3. Create Equipment nodes and relationships
        for tag in equipment_tags:
            await self.neo4j.upsert_equipment(tag_id=tag)
            await self.neo4j.create_relationship(
                "Equipment", "tag_id", tag,
                "Document", "doc_id", doc_id,
                "HAS_DOCUMENT",
            )
            logger.debug(f"Linked Equipment {tag} -> Document {doc_id}")

        # 4. Create Regulation nodes and relationships
        for ref in regulatory_refs:
            await self.neo4j.upsert_regulation(code=ref)
            await self.neo4j.create_relationship(
                "Document", "doc_id", doc_id,
                "Regulation", "code", ref,
                "REFERENCES",
            )
            # Also link equipment to regulations
            for tag in equipment_tags:
                await self.neo4j.create_relationship(
                    "Equipment", "tag_id", tag,
                    "Regulation", "code", ref,
                    "SUBJECT_TO",
                )

        # 5. Create Person nodes
        for person_str in persons:
            # Parse "Name (Role)" format
            name = person_str.split("(")[0].strip()
            role = ""
            if "(" in person_str and ")" in person_str:
                role = person_str.split("(")[1].rstrip(")")
            await self.neo4j.upsert_person(name=name, role=role)

        # 6. Create WorkOrder nodes and link to equipment
        for wo_id in work_orders:
            await self.neo4j.upsert_work_order(wo_id=wo_id)
            for tag in equipment_tags:
                await self.neo4j.create_relationship(
                    "WorkOrder", "wo_id", wo_id,
                    "Equipment", "tag_id", tag,
                    "PERFORMED_ON",
                )

        logger.info(
            f"Graph built for doc {doc_id}: "
            f"{len(equipment_tags)} equipment, {len(regulatory_refs)} regulations, "
            f"{len(persons)} persons, {len(work_orders)} work orders"
        )

    async def build_incident_node(
        self,
        incident_id: str,
        description: str,
        equipment_tag: str,
        severity: str = "MEDIUM",
        date: str = "",
        root_cause: str = "",
        resolution: str = "",
    ):
        """Create an Incident node and link it to equipment."""
        await self.neo4j.upsert_incident(
            incident_id=incident_id,
            description=description,
            severity=severity,
            date=date,
            root_cause=root_cause,
            resolution=resolution,
        )

        if equipment_tag:
            await self.neo4j.upsert_equipment(tag_id=equipment_tag)
            await self.neo4j.create_relationship(
                "Incident", "incident_id", incident_id,
                "Equipment", "tag_id", equipment_tag,
                "INVOLVES",
            )

    async def link_similar_incidents(
        self,
        incident_id_1: str,
        incident_id_2: str,
        similarity_score: float,
    ):
        """Create a SIMILAR_TO relationship between two incidents."""
        await self.neo4j.create_relationship(
            "Incident", "incident_id", incident_id_1,
            "Incident", "incident_id", incident_id_2,
            "SIMILAR_TO",
            properties={"similarity_score": similarity_score},
        )
