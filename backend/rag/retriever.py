"""
FILE PURPOSE
------------
Schema retrieval with RELATIONSHIP EXPANSION.

PIPELINE
--------
  question -> embed -> vector similarity search (top-K chunks)
           -> relationship expansion (add "bridge" tables between hits)
           -> relevant schema text for the LLM prompt

WHY EXPANSION?  "Find the average rating of each vehicle" mentions `rating` and
`vehicle`, but the SQL must go rating -> rental -> vehicles. The word "rental"
is absent, so pure similarity may rank it low. We therefore build an undirected
graph from foreign keys and, for every pair of the top-3 "anchor" hits, add the
tables on the shortest FK path between them (max 3 hops).

INPUT  : question string, top_k.
OUTPUT : `RetrievalResult` (ranked tables with scores, expanded/bridge tables, final
         table list, schema text incl. relationships).
CALLED BY: tools/retrieval_tool.py, evaluation/*, tests.
CALLS  : rag/embeddings.py, rag/vector_store.py, database/schema_inspector.py.
EXAMPLE: retrieve("average rating of vehicles").tables -> ['rating','vehicles','rental',...]
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from backend.config import Settings, get_settings
from backend.rag.embeddings import EmbeddingProvider, get_embedder
from backend.rag.vector_store import VectorStore
from backend.schemas import DatabaseSchema


@dataclass
class RetrievalResult:
    ranked: list[tuple[str, float]] = field(default_factory=list)   # raw similarity ranking (top-k)
    expanded: list[str] = field(default_factory=list)               # bridge tables added
    tables: list[str] = field(default_factory=list)                 # final list given to the LLM
    schema_text: str = ""


def _fk_graph(schema: DatabaseSchema) -> dict[str, set[str]]:
    g: dict[str, set[str]] = {t: set() for t in schema.table_names}
    for t, _c, rt, _rc in schema.relationships():
        if rt in g:
            g[t].add(rt)
            g[rt].add(t)
    return g


def shortest_path(g: dict[str, set[str]], a: str, b: str, max_hops: int = 3) -> list[str]:
    if a == b:
        return [a]
    seen, q = {a}, deque([[a]])
    while q:
        path = q.popleft()
        if len(path) - 1 >= max_hops:
            continue
        for n in g.get(path[-1], ()):
            if n in seen:
                continue
            if n == b:
                return path + [n]
            seen.add(n)
            q.append(path + [n])
    return []


class SchemaRetriever:
    def __init__(self, schema: DatabaseSchema, store: VectorStore,
                 embedder: EmbeddingProvider | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.schema, self.store = schema, store
        self.embedder = embedder or get_embedder(self.settings)
        self._graph = _fk_graph(schema)
        self._chunk_text = {i["table"]: i["text"] for i in store.items}

    def retrieve(self, question: str, top_k: int | None = None, expand: bool = True) -> RetrievalResult:
        k = top_k or self.settings.top_k
        qv = self.embedder.embed([question], is_query=True)[0]
        hits = self.store.search(qv, k)
        ranked = [(h["table"], round(score, 4)) for score, h in hits]
        tables = [t for t, _ in ranked]
        bridges: list[str] = []
        if expand:
            # anchors = strongest hits (top-3, and at least half as similar as the best hit)
            best = ranked[0][1] if ranked else 0.0
            anchors = [t for t, sc in ranked[:3] if sc >= 0.5 * best]
            for i in range(len(anchors)):
                for j in range(i + 1, len(anchors)):
                    for node in shortest_path(self._graph, anchors[i], anchors[j]):
                        if node not in tables and node not in bridges:
                            bridges.append(node)
        final = tables + bridges
        return RetrievalResult(ranked=ranked, expanded=bridges, tables=final,
                               schema_text=self.render_schema(final))

    def render_schema(self, tables: list[str]) -> str:
        parts = [self._chunk_text[t] for t in tables if t in self._chunk_text]
        sel = {t.lower() for t in tables}
        rels = [f"{t}.{c} = {rt}.{rc}" for (t, c, rt, rc) in self.schema.relationships()
                if t.lower() in sel and rt.lower() in sel]
        text = "\n\n".join(parts)
        if rels:
            text += "\n\nRelationships (valid join conditions):\n" + "\n".join(f"- {r}" for r in rels)
        return text
