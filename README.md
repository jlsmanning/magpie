# Magpie

AI-powered research paper discovery system with conversational interest management.

**Status:** Work in Progress

## Overview

Magpie helps researchers discover relevant papers using RAG (Retrieval-Augmented Generation) with natural language interest management. Instead of manually searching databases, you tell Magpie what you're interested in and it finds papers for you.

## Planned Features

- [ ] Multi-query semantic search using RAG
- [ ] Natural language profile management (add/remove research interests)
- [x] ArXiv paper ingestion and indexing
- [ ] LLM-powered result synthesis and explanations
- [ ] Zotero integration for saving papers
- [ ] Serendipity sampler (cross-field recommendations)
- [ ] Voice/audio interface

## Implementation Status

### Core Components
- [ ] Input Parser/Manager
- [x] Query Processor
- [x] Vector Search
- [ ] Reranker
- [ ] Synthesizer
- [ ] Profile Manager

### Integrations
- [ ] ArXiv Puller
- [x] Embedder (Sentence-BERT)
- [ ] LLM Client (Claude)
- [ ] Zotero Client

### Data Models
- [x] Paper schema
- [x] Query schema
- [x] Profile schema
- [x] Results schema

## Tech Stack

- **Vector DB**: ChromaDB (local)
- **Embeddings**: Sentence-Transformers (local)
- **LLM**: Claude API
- **Data Source**: ArXiv API
- **Language**: Python 3.10+

## Setup (when ready)

## License

MIT