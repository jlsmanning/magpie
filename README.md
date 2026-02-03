# Magpie
AI-powered research paper discovery system with conversational query building and interactive review.

**Status:** Work in Progress - Core pipeline functional, testing in progress

## Overview
Magpie helps researchers discover relevant papers using RAG (Retrieval-Augmented Generation) and natural language conversation. Instead of manually searching databases with keywords, you have a conversation with Magpie to build queries, then review papers interactively with AI assistance.

**Key Innovation:** Designed for voice-first interaction - review papers while walking, driving, or doing other activities.

## Features
- [x] **Conversational query building** - Build search queries through natural dialogue
- [x] **Multi-query semantic search** - Handles complex, multi-topic queries with independent evaluation
- [x] **LLM-powered reranking** - Deep relevance evaluation beyond keyword matching
- [x] **Interactive review** - Conversational exploration of papers with on-demand PDF analysis
- [x] **Zotero integration** - Save papers directly to your reference library
- [x] **Profile persistence** - Maintains research context and search history
- [ ] **Serendipity sampler** - Cross-field recommendations (planned)
- [ ] **Voice/audio interface** - Hands-free paper review (planned)

## Tech Stack
- **Vector DB**: ChromaDB (local)
- **Embeddings**: Sentence-BERT (`all-mpnet-base-v2`)
- **LLM**: Claude API (Anthropic)
- **Data Source**: ArXiv API
- **Reference Manager**: Zotero (optional)
- **Language**: Python 3.10+

## Setup

### 1. Install Dependencies
```bash
# Create conda environment
conda create -n magpie python=3.10
conda activate magpie

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
```

**Required:**
- `ANTHROPIC_API_KEY` - Get from https://console.anthropic.com/

**Optional (for Zotero integration):**
- `ZOTERO_LIBRARY_ID` - Your User ID from https://www.zotero.org/settings/keys
- `ZOTERO_API_KEY` - Create new API key with library, write, and file access
- `ZOTERO_COLLECTION_NAME` - Collection name (default: "magpie")

### 3. Populate Paper Database
```bash
# Index papers from ArXiv (one-time setup, ~15 minutes)
python scripts/populate_papers.py --categories cs.AI cs.LG cs.CV --years 2 --max-results 1000

# Or just recent papers for testing (~5 minutes)
python scripts/populate_papers.py --categories cs.AI --recent 7 --max-results 50
```

This downloads papers from ArXiv, generates embeddings, and stores them in ChromaDB at `./data/vector_db/`.

## Usage

### Start Magpie
```bash
python cli/main.py
```

### Build a Query

Magpie uses conversational AI to help you build queries:
```
> I want papers on transformers
I've added 'transformers' to your query. Ready to search, or want to add more topics?

> Also add attention mechanisms
I've added 'attention mechanisms'. Both are weighted equally at 50%. 

> Make transformers more important
I've increased the weight for 'transformers' to 60%.

> Run search
```

### Review Results

After search completes, Magpie displays results and asks if you want to review:
```
Would you like to review these papers interactively? (yes/no)
> yes

Paper 1 of 8: Attention Is All You Need
This paper introduces the Transformer architecture...

Would you like to know more, skip it, or save to Zotero?
> Tell me about the methods
[Downloads PDF if needed]
The paper proposes a novel architecture that relies entirely on attention mechanisms...

> Does it have code available?
[Scans PDF]
Yes, the authors mention an implementation in the appendix at github.com/...

> Save it
Saved to Zotero! Moving to paper 2...
```

### Research Context

Set your research context for better personalized results:
```
> Update my context: I'm a PhD student studying fairness in computer vision. I prefer recent papers with code.
I've updated your research context. This will help me suggest more relevant papers.
```

### Commands

- **Build queries:** "I want papers on X", "Also add Y", "Make X more important"
- **Execute search:** "Run search", "Find papers", "Search"
- **Review papers:** "Tell me more", "Does it have code?", "Read the methods"
- **Save/skip:** "Save it", "Save to Zotero", "Skip", "Next paper"
- **Navigate:** "Go to paper 3", "Go back", "Show my query"
- **Exit:** "exit", "quit"

## Project Structure
```
magpie/
├── cli/                 # Command-line interface
├── magpie/
│   ├── core/           # Pipeline components (query, search, rerank, synthesize, review)
│   ├── data/           # Data ingestion (ArXiv puller, paper indexer)
│   ├── integrations/   # External services (LLM, embedder, Zotero, PDF fetcher)
│   ├── models/         # Data models (Paper, Query, Profile, Results)
│   └── utils/          # Configuration and profile management
├── scripts/            # Utility scripts (populate database)
├── data/
│   ├── profiles/       # User profiles (gitignored)
│   ├── vector_db/      # ChromaDB storage (gitignored)
│   └── pdfs/           # Downloaded PDFs (gitignored)
└── requirements.txt
```

## Roadmap

### Current (Phase 1)
- [x] Complete pipeline: query building → search → rerank → synthesize → review
- [x] Zotero integration
- [x] PDF analysis for deep questions

### Next (Phase 2)
- [ ] Docker containerization
- [ ] REST API
- [ ] Cloud deployment (AWS/GCP)
- [ ] CI/CD pipeline
- [ ] Automated paper ingestion (scheduled updates)

### Future (Phase 3)
- [ ] Voice/audio interface for hands-free review
- [ ] Serendipity sampler (cross-field recommendations)
- [ ] Multi-source paper retrieval (Semantic Scholar, PubMed)
- [ ] Citation graph exploration

## Development

### Running Tests
```bash
pytest tests/
```

### Updating Paper Database
```bash
# Add recent papers (run weekly/monthly)
python scripts/populate_papers.py --categories cs.AI cs.LG --recent 7
```

## Troubleshooting

**"No papers in database"**
- Run `python scripts/populate_papers.py` first (see Setup step 3)

**"Zotero not configured"**
- Set `ZOTERO_LIBRARY_ID` and `ZOTERO_API_KEY` in `.env`
- Or skip Zotero integration (papers won't be saved)

**"ANTHROPIC_API_KEY not set"**
- Get API key from https://console.anthropic.com/
- Add to `.env` file

## License
MIT