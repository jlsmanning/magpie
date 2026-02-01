# Magpie

AI-powered research paper discovery system with conversational interest management.

**Status:** Work in Progress

## Overview

Magpie helps researchers discover relevant papers using RAG (Retrieval-Augmented Generation) with natural language interest management. Instead of manually searching databases, you tell Magpie what you're interested in and it finds papers for you.

## Planned Features

- [x] Multi-query semantic search using RAG
- [x] Conversational query-building with LLM
- [ ] Profile persistence (save/load user research context and queries) 
- [x] ArXiv paper ingestion and indexing
- [x] LLM-powered result synthesis and explanations
- [ ] Zotero integration for saving papers
- [ ] Serendipity sampler (cross-field recommendations)
- [ ] Voice/audio interface

## Implementation Status

### Core Components
- [x] Input Parser/Manager
- [x] Query Processor
- [x] Vector Search
- [x] Reranker
- [x] Synthesizer
- [x] Profile Manager
- [ ] Interactive Reviewer

### Integrations
- [x] ArXiv Puller
- [x] Embedder (Sentence-BERT)
- [x] LLM Client (Claude)
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