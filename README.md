<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Claude-claude--sonnet--4--6-blueviolet?style=for-the-badge&logo=anthropic&logoColor=white"/>
<img src="https://img.shields.io/badge/LlamaIndex-RAG-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Neo4j-Knowledge_Graph-008CC1?style=for-the-badge&logo=neo4j&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ET_AI_Hackathon-2026_PS8-gold?style=for-the-badge"/>

# 🧠 KnowledgeBrain API

### *AI for Industrial Knowledge Intelligence — Unified Asset & Operations Brain*

> **One unified intelligence layer across all your industrial documents.**
> Query maintenance records, engineering drawings, compliance procedures, and incident reports in natural language — in seconds, not hours.

[🚀 Quick Start](#-quick-start) · [📐 Architecture](#-architecture) · [🔌 API Reference](#-api-reference) · [📊 Evaluation](#-evaluation-metrics) · [👥 Team](#-team)

</div>

---

## 📌 The Problem

Industrial facilities in India and globally are drowning in documents — yet paradoxically starved of actionable knowledge at the moment decisions need to be made.

| Statistic | Source |
|-----------|--------|
| **35%** of industrial professionals' working hours lost to information search | McKinsey 2024 |
| **7–12** disconnected document systems in the average large Indian plant | NASSCOM-EY |
| **18–22%** of unplanned downtime events attributable to knowledge fragmentation | BIS Research |
| **25%** of experienced engineers retiring within 10 years, taking tacit knowledge with them | Industry estimate |

A maintenance technician diagnosing a pump failure cannot instantly cross-reference the OEM manual, the last three inspection reports, the work order history, and the relevant OISD procedure in a single query. Each document lives in a different system, in a different format, accessible only to a different department. **KnowledgeBrain solves this.**

---

## ✨ Features

### 1. 📄 Universal Document Ingestion & Knowledge Graph
Ingest every document type found in an industrial facility into a single queryable intelligence layer:
- **PDFs** — engineering specs, MSDS, regulatory submissions, operating manuals
- **P&IDs** — plant and instrumentation diagrams parsed via computer vision (YOLOv8)
- **Scanned forms** — inspection records, permit-to-work logs, maintenance sign-offs (OCR via Azure Document Intelligence)
- **Spreadsheets** — maintenance history, calibration records, spare parts inventories
- **Email archives** — informal operational knowledge, incident alerts, project correspondence

Entities (equipment tags, process parameters, regulatory references, personnel, dates, failure codes) are extracted and connected in a **Neo4j property knowledge graph** that auto-updates as new documents arrive.

---

### 2. 🤖 Expert Knowledge Copilot
A RAG-powered conversational AI that answers operational, maintenance, and engineering queries against the full document corpus:
- **Source-cited** — every answer links to originating documents with page references
- **Confidence-scored** — communicates uncertainty, never false certainty
- **Multi-language** — supports English, Hindi (हिन्दी), Telugu (తెలుగు), Tamil (தமிழ்)
- **Offline-capable** — core query works with local model cache in low-connectivity zones
- **Mobile-first** — field technicians query in the field, not engineers at desktops

**Example queries handled:**
```
"What is the last recorded valve torque for V-204B and when was it calibrated?"
"Show me the lockout/tagout procedure for the boiler feed pump at Train 2."
"What were the findings from the last OISD audit and which NCRs are still open?"
"What caused the similar vibration issue on P-101A in 2022 and what was the fix?"
```

---

### 3. 🔧 Maintenance Intelligence & RCA Agent
An agentic system (LangGraph) that fuses work order history, equipment failure records, OEM manuals, and real-time readings to:
- Generate **predictive maintenance recommendations** based on failure pattern analysis
- Provide structured **Root Cause Analysis (RCA)** using the 5-Whys methodology
- Optimise maintenance schedules against operational windows
- Surface **cross-equipment failure patterns** invisible to individual maintenance teams
- Reduce **MTTR** by pre-staging the right information before the technician arrives

---

### 4. 🛡️ Quality & Regulatory Compliance Intelligence
An agentic compliance layer mapping regulatory requirements (Factory Act, OISD, PESO, BIS) against current procedures and equipment states:
- **Continuously identifies compliance gaps** before audits find them
- **Auto-generates** compliance evidence packages for Factory Inspector or OISD audits
- **Flags quality deviations** in real time and routes to the responsible owner
- **Live regulatory change tracker** — when OISD publishes an update, affected procedures are flagged immediately

---

### 5. 📚 Lessons Learned & Failure Intelligence Engine
Institutional memory across the organisation's entire history of incidents and near-misses:
- Identifies **systemic patterns** no single investigation would detect
- **Proactively pushes warnings** when current conditions match pre-failure patterns
- Preserves **tacit knowledge of retiring engineers** by structuring their incident narratives into the knowledge graph
- Delivers a **"similar incidents" briefing** the moment a new incident is logged

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            LAYER 5 — Multi-Channel Interface                │
│         Web (React)  ·  Mobile (React Native)  ·  API       │
├─────────────────────────────────────────────────────────────┤
│               LAYER 4 — Application Layer                   │
│   Copilot  ·  Maintenance Agent  ·  Compliance Agent        │
│                   ·  Lessons Learned Engine                 │
├─────────────────────────────────────────────────────────────┤
│            LAYER 3 — RAG + Agentic Reasoning                │
│       LlamaIndex  ·  Qdrant  ·  LangGraph Agents            │
├─────────────────────────────────────────────────────────────┤
│          LAYER 2 — Knowledge Graph & Entity Store           │
│      Neo4j Property Graph  ·  Equipment-Procedure-Risk      │
├─────────────────────────────────────────────────────────────┤
│         LAYER 1 — Document Ingestion & Processing           │
│   PDF  ·  P&ID (YOLOv8)  ·  OCR  ·  Spreadsheet  ·  Email  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
Raw Documents ──► Ingestion Pipeline ──► Chunking & Entity Extraction
                                                │
                              ┌─────────────────┴─────────────────┐
                              ▼                                   ▼
                        Qdrant Vector DB                   Neo4j Graph DB
                        (semantic search)              (entity relationships)
                              │                                   │
                              └─────────────────┬─────────────────┘
                                                │
                                    Hybrid Retrieval Engine
                                                │
                                    Claude claude-sonnet-4-6 (LLM)
                                                │
                                     Structured Response
                              (answer + sources + confidence + citations)
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM Backend** | Claude `claude-sonnet-4-6` (Anthropic) | Primary reasoning, RAG synthesis, multi-language |
| **RAG Framework** | LlamaIndex | Document ingestion pipelines, chunking, retrieval |
| **Vector Database** | Qdrant | High-performance semantic search |
| **Knowledge Graph** | Neo4j | Equipment-document-risk entity relationships |
| **OCR & Doc Intel** | Azure Document Intelligence | Scanned PDFs, forms, P&IDs |
| **Computer Vision** | YOLOv8 (custom fine-tune) | P&ID symbol detection and tag extraction |
| **Agentic Framework** | LangGraph | Multi-agent orchestration |
| **Backend API** | FastAPI (Python 3.11) | REST + WebSocket API layer |
| **Frontend** | React + TailwindCSS | Engineer and manager dashboards |
| **Mobile** | React Native (Expo) | Field technician mobile interface |
| **Infrastructure** | Docker + Kubernetes | Container-native, cloud-agnostic deployment |
| **Storage** | MinIO + PostgreSQL + Redis | Documents, metadata, caching |
| **Auth** | Keycloak | Role-based access control |

---

## 📁 Project Structure

```
KnowledgeBrainAPI/
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── config.py                   # Settings via pydantic-settings
│   ├── routers/
│   │   ├── ingest.py               # POST /ingest
│   │   ├── query.py                # POST /query
│   │   ├── copilot.py              # POST /copilot/chat
│   │   ├── maintenance.py          # POST /maintenance/predict, /rca
│   │   ├── compliance.py           # POST /compliance/gaps, /audit-package
│   │   ├── lessons.py              # POST /lessons/analyze-incident
│   │   └── health.py               # GET  /health
│   ├── services/
│   │   ├── ingestion/
│   │   │   ├── pdf_processor.py
│   │   │   ├── pid_processor.py    # YOLOv8 P&ID parser
│   │   │   ├── spreadsheet_processor.py
│   │   │   └── email_processor.py
│   │   ├── rag/
│   │   │   ├── embedder.py
│   │   │   ├── retriever.py        # Hybrid semantic + graph retrieval
│   │   │   └── synthesizer.py      # Claude-powered answer generation
│   │   ├── graph/
│   │   │   ├── neo4j_client.py
│   │   │   ├── entity_extractor.py
│   │   │   └── graph_builder.py
│   │   └── agents/
│   │       ├── maintenance_agent.py   # LangGraph agent
│   │       ├── compliance_agent.py    # LangGraph agent
│   │       └── lessons_agent.py       # LangGraph agent
│   └── models/
│       ├── document.py
│       ├── query.py
│       └── alert.py
├── frontend/                       # React + TailwindCSS web app
├── mobile/                         # React Native (Expo) field app
├── eval/
│   ├── benchmark_questions.json    # 20 domain-expert questions
│   ├── ground_truth.json           # Entity extraction ground truth
│   ├── test_entity_extraction.py
│   ├── test_query_quality.py
│   ├── test_graph_linkage.py
│   ├── test_performance.py
│   └── run_eval.py                 # Master evaluation runner
├── data/
│   └── synthetic/                  # Demo dataset (OISD reports, sample P&IDs)
├── scripts/
│   └── generate_demo_data.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- An [Anthropic API key](https://console.anthropic.com)
- Azure Document Intelligence resource (for OCR)

### 1. Clone the repository

```bash
git clone https://github.com/SakshamDevloper/KnowledgeBrainAPI.git
cd KnowledgeBrainAPI
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys (see Environment Variables below)
```

### 3. Run with Docker Compose (recommended)

```bash
docker-compose up --build
```

This starts: FastAPI backend · Qdrant · Neo4j · Redis · MinIO

App will be available at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### 4. Run for development (without Docker)

```bash
# Backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                        # http://localhost:3000
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
# ── Anthropic ──────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ── Qdrant ─────────────────────────────────────
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=knowledgebrain_docs

# ── Neo4j ──────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# ── Azure Document Intelligence ────────────────
AZURE_FORM_RECOGNIZER_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_FORM_RECOGNIZER_KEY=your_azure_key

# ── Redis ──────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6379

# ── MinIO (Object Storage) ─────────────────────
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=knowledgebrain

# ── App Settings ───────────────────────────────
APP_ENV=development
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE_MB=100
DEFAULT_TOP_K=10
CONVERSATION_TTL_HOURS=24
```

---

## 🔌 API Reference

### Document Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest/pdf` | Ingest a PDF document |
| `POST` | `/ingest/spreadsheet` | Ingest Excel/CSV maintenance data |
| `POST` | `/ingest/email` | Ingest email archive (.eml / .txt) |
| `POST` | `/ingest/batch` | Batch ingest a folder of documents |
| `GET`  | `/ingest/status/{job_id}` | Check ingestion job status |

### Copilot Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/copilot/chat` | Send a query, receive answer + sources |
| `GET`  | `/copilot/history/{conversation_id}` | Retrieve conversation history |
| `DELETE` | `/copilot/history/{conversation_id}` | Clear conversation |

**Example request:**
```json
POST /copilot/chat
{
  "message": "What caused the vibration failure on P-101A in 2022?",
  "language": "en",
  "equipment_context": ["P-101A"],
  "conversation_id": "optional-session-id"
}
```

**Example response:**
```json
{
  "answer": "The 2022 vibration failure on P-101A was caused by...",
  "sources": [
    {"filename": "RCA_P101A_2022.pdf", "page": 4, "relevance": 0.94}
  ],
  "confidence": "HIGH",
  "follow_up_suggestions": ["What was the resolution?", "Is this failure mode common?"],
  "safety_flag": false,
  "conversation_id": "conv_abc123"
}
```

### Maintenance Intelligence

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/maintenance/predict` | Predictive maintenance for an equipment tag |
| `POST` | `/maintenance/rca` | Run root cause analysis for an incident |
| `GET`  | `/maintenance/schedule/{equipment_tag}` | Get maintenance schedule |

### Compliance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/compliance/gaps` | Detect compliance gaps (all or by equipment) |
| `POST` | `/compliance/audit-package` | Generate audit evidence package |
| `POST` | `/compliance/check-regulation-change` | Check impact of a regulation update |

### Lessons Learned

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/lessons/analyze-incident` | Analyze incident, find similar past events |
| `POST` | `/lessons/capture-expert-knowledge` | Capture retiring engineer's knowledge |
| `GET`  | `/lessons/alerts` | Get active proactive alerts |

---

## 📊 Evaluation Metrics

Run the full evaluation suite:

```bash
cd eval
python run_eval.py
# Outputs: evaluation_summary.json
```

| Metric | Target | How Measured |
|--------|--------|-------------|
| **Entity Extraction Accuracy** | > 90% F1 | Equipment tags, process params, regulatory refs across 5 doc types |
| **Query Answer Quality** | > 85% relevance | Claude-as-judge on 20 domain-expert benchmark questions |
| **Knowledge Graph Linkage** | > 80% | Entities correctly cross-linked across document types in demo corpus |
| **Time-to-Answer** | < 5 seconds (P95) | End-to-end query against 10,000-document corpus |
| **Compliance Gap Detection** | > 88% precision | On known-gap test set from synthetic regulatory corpus |

---

## 🎬 Demo Scenario

**Narrative:** A centrifugal pump (P-101A) at a petroleum refinery shows early vibration signatures.

**KnowledgeBrain's response in under 90 seconds:**

1. **Copilot** instantly surfaces the last 3 maintenance records, OEM vibration thresholds, and the last inspection report — in a single query with source citations.
2. **Maintenance Agent** cross-references similar past failures and recommends bearing inspection as the primary hypothesis, backed by two historical RCAs from 2021 and 2023.
3. **Compliance Agent** flags that an OISD-18 inspection is 47 days overdue and auto-drafts the compliance evidence package.
4. **Lessons Learned Engine** surfaces a 2019 incident where the same failure signature preceded a seal failure — and proactively alerts the shift supervisor.

> The same workflow without KnowledgeBrain typically takes **2–4 hours** of manual search across 7+ systems.

---

## 🗂️ Demo Dataset

All data is publicly available — no licensing issues:

| Dataset | Source | Used For |
|---------|--------|----------|
| OISD Incident Investigation Reports | [oisd.gov.in](http://oisd.gov.in) (public) | Lessons Learned Engine |
| Sample P&IDs | ISA training resources | Computer vision pipeline |
| Factory Act text | [India Code portal](https://www.indiacode.nic.in) | Compliance corpus |
| PESO regulations | [peso.gov.in](http://peso.gov.in) | Compliance corpus |
| Synthetic work orders | `scripts/generate_demo_data.py` | Maintenance agent |

Generate synthetic demo data:
```bash
python scripts/generate_demo_data.py --equipment-count 50 --years 5
```

---

## 🏆 Hackathon Context

**Event:** ET AI Hackathon 2026
**Problem Statement:** PS 8 — AI for Industrial Knowledge Intelligence
**Judging Criteria:**

| Criterion | Weight | Our Approach |
|-----------|--------|-------------|
| Innovation | 25% | Multi-modal knowledge graph across P&IDs, scanned forms, emails — proactive failure push |
| Business Impact | 25% | 35% of wasted hours recoverable; 18–22% downtime reduction; audit prep weeks → hours |
| Technical Excellence | 20% | Multi-agent LangGraph; hybrid retrieval; CV P&ID parsing; confidence-scored answers |
| Scalability | 15% | Container-native; Qdrant + Neo4j horizontally scalable; SaaS-ready for 100M+ docs |
| User Experience | 15% | Mobile-first; sub-5s response; conversational; multi-language; offline-capable |

---

## 👥 Team

| Name | Role | Responsibilities |
|------|------|-----------------|
| **Saksham Sethi** | Lead Architect | RAG pipeline, LlamaIndex integration, Qdrant vector store, API layer |
| **Sumit Singh Charak** | AI/ML Engineer | LLM prompt engineering, LangGraph agents, Maintenance & Compliance agents, evaluation |
| **Gummadidala Sai Krishna** | Full-stack & Infra | React/React Native UI, Neo4j knowledge graph, OCR/CV pipeline, deployment |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. Please ensure tests pass before submitting:

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- McKinsey Global Survey on Industrial Information Access, 2024
- NASSCOM-EY Manufacturing & Energy Digital Maturity Study, 2024
- BIS Research: Industrial Downtime Cost Analysis, Indian Heavy Industry, 2024
- [Anthropic API Documentation](https://docs.anthropic.com)
- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [Neo4j Knowledge Graph Docs](https://neo4j.com/docs)

---

<div align="center">

**Built with ❤️ for ET AI Hackathon 2026 · PS 8**

*Making every piece of industrial knowledge queryable — instantly, accurately, at the moment it's needed.*

</div>
