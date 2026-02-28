import logging
import json
import os
from typing import Any, Dict, List, Optional

from .game_tools import ResultEvaluator, VectorDBTool, WebSearchTool

logging.basicConfig(level=logging.INFO)


class WorkflowStateMachine:
    END = "__END__"

    def __init__(self):
        self._nodes = {}

    def add_node(self, name, handler):
        self._nodes[name] = handler

    def run(self, start_node: str, state: Dict[str, Any]) -> Dict[str, Any]:
        current = start_node
        steps = 0
        while current != self.END:
            if current not in self._nodes:
                raise RuntimeError("Unknown workflow node: %s" % current)
            if steps > 20:
                raise RuntimeError("Workflow exceeded max steps")
            self._nodes[current](state)
            current = state.get("next_node", self.END)
            steps += 1
        return state


class GameAgent:
    def __init__(
        self,
        vector_tool: VectorDBTool,
        evaluator: ResultEvaluator,
        web_tool: Optional[WebSearchTool] = None,
        model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
    ):
        self.vector_tool = vector_tool
        self.evaluator = evaluator
        self.web_tool = web_tool
        self.model = model
        self.history = []
        self._client = None
        api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            except Exception as exc:  # pragma: no cover
                logging.warning("OpenAI client unavailable: %s", exc)
        self.workflow = self._build_workflow()

    def _handle_llm_error(self, exc: Exception):
        message = str(exc).lower()
        if any(token in message for token in ("insufficient_quota", "rate limit", "error code: 429")):
            logging.warning("Disabling LLM after quota/rate-limit error.")
            self._client = None

    def _build_workflow(self):
        wf = WorkflowStateMachine()
        wf.add_node("rewrite_query", self._node_rewrite_query)
        wf.add_node("retrieve", self._node_retrieve)
        wf.add_node("decide", self._node_decide)
        wf.add_node("web_search", self._node_web_search)
        wf.add_node("synthesize", self._node_synthesize)
        return wf

    def _history_context(self, turns: int = 4):
        recent = self.history[-turns:]
        items = []
        for i, turn in enumerate(recent, start=1):
            result = turn.get("result", {})
            items.append(
                {
                    "turn": i,
                    "query": turn.get("query", ""),
                    "answer": result.get("answer", ""),
                    "source": result.get("source", ""),
                }
            )
        return items

    def _heuristic_rewrite_query(self, query: str):
        q = query.lower()
        if not self.history:
            return query
        last = self.history[-1].get("result", {})
        results = last.get("results", [])
        first_item = results[0] if results else {}
        meta = first_item.get("metadata", {}) if isinstance(first_item, dict) else {}
        game_name = meta.get("Name")
        if not game_name and last.get("citations"):
            game_name = last["citations"][0].get("label", "").split("(")[0].strip()
        if not game_name:
            return query

        if "first game" in q or "first one" in q or "that game" in q or "it " in q:
            return "Tell me more about %s" % game_name
        return query

    def _build_citations(self, source: str, retrieved: List[Dict[str, Any]], web_results: List[Dict[str, Any]]):
        citations = []
        if source == "internal_db":
            for i, item in enumerate(retrieved[:3], start=1):
                meta = item.get("metadata", {})
                citations.append(
                    {
                        "id": str(i),
                        "label": "%s (%s)" % (meta.get("Name", "Unknown"), meta.get("YearOfRelease", "N/A")),
                        "source": "internal_db",
                    }
                )
        elif source == "web_search":
            for i, item in enumerate(web_results[:3], start=1):
                citations.append(
                    {
                        "id": str(i),
                        "label": item.get("title", "Result %d" % i),
                        "source": item.get("url", ""),
                    }
                )
        return citations

    def _node_rewrite_query(self, state: Dict[str, Any]):
        query = state["query"]
        state["effective_query"] = self._heuristic_rewrite_query(query)
        if not self._client or not self.history:
            state["next_node"] = "retrieve"
            return
        prompt = (
            "Rewrite the user query into a standalone query using chat history.\n"
            "If not needed, return the original query.\n"
            "Return strict JSON: {\"query\": \"...\"}."
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"query": query, "history": self._history_context()},
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            rewritten = json.loads(response.choices[0].message.content or "{}").get("query", state["effective_query"])
            state["effective_query"] = rewritten
        except Exception as exc:
            self._handle_llm_error(exc)
            pass
        state["next_node"] = "retrieve"

    def _node_retrieve(self, state: Dict[str, Any]):
        logging.info("Using internal DB retrieval...")
        state["retrieved"] = self.vector_tool.retrieve(
            state["effective_query"],
            n_results=state["n_results"],
        )
        state["next_node"] = "decide"

    def _node_decide(self, state: Dict[str, Any]):
        retrieved = state["retrieved"]
        if not self._client:
            use_web = not self.evaluator.is_sufficient(retrieved)
            state["reasoning"] = "No LLM configured; used threshold-based evaluator."
            state["use_web"] = use_web
            state["next_node"] = "web_search" if use_web else "synthesize"
            return

        preview = []
        for item in retrieved[:3]:
            meta = item.get("metadata", {})
            preview.append(
                {
                    "distance": item.get("distance"),
                    "name": meta.get("Name"),
                    "platform": meta.get("Platform"),
                    "publisher": meta.get("Publisher"),
                    "year": meta.get("YearOfRelease"),
                    "document": item.get("document", "")[:240],
                }
            )
        prompt = (
            "Decide if web search is needed.\n"
            "Use web if internal results are irrelevant, missing, or likely outdated.\n"
            "Return strict JSON: {\"use_web\": boolean, \"reasoning\": string}."
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "query": state["query"],
                                "effective_query": state["effective_query"],
                                "history": self._history_context(),
                                "retrieved_preview": preview,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            state["use_web"] = bool(parsed.get("use_web", False))
            state["reasoning"] = str(parsed.get("reasoning", "No reasoning provided."))
        except Exception as exc:
            logging.warning("LLM decision failed; fallback evaluator used: %s", exc)
            self._handle_llm_error(exc)
            state["use_web"] = not self.evaluator.is_sufficient(retrieved)
            state["reasoning"] = "LLM decision failed; used threshold-based evaluator."
        state["next_node"] = "web_search" if state["use_web"] else "synthesize"

    def _node_web_search(self, state: Dict[str, Any]):
        state["web_results"] = []
        if self.web_tool is None:
            state["source"] = "none"
            state["next_node"] = "synthesize"
            return
        logging.info("Falling back to Tavily web search...")
        state["web_results"] = self.web_tool.search(
            state["effective_query"],
            n_results=state["n_results"],
        )
        state["source"] = "web_search" if state["web_results"] else "none"
        state["next_node"] = "synthesize"

    def _synthesize_with_llm(self, state: Dict[str, Any], citations: List[Dict[str, str]]):
        if not self._client:
            return None
        payload = {
            "query": state["query"],
            "effective_query": state["effective_query"],
            "history": self._history_context(),
            "source": state["source"],
            "retrieved": state["retrieved"][:3],
            "web_results": state.get("web_results", [])[:3],
            "citations": citations,
        }
        prompt = (
            "Write a concise final answer for the user based only on provided evidence.\n"
            "Use inline citations exactly as [1], [2], etc based on the provided citation ids.\n"
            "If evidence is weak, explicitly say uncertainty.\n"
            "Do not output JSON."
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if citations and "[" not in text:
                text += " [1]"
            return text
        except Exception as exc:
            logging.warning("LLM synthesis failed: %s", exc)
            self._handle_llm_error(exc)
            return None

    def _fallback_answer(self, state: Dict[str, Any], citations: List[Dict[str, str]]):
        source = state["source"]
        retrieved = state["retrieved"]
        web_results = state.get("web_results", [])
        if source == "internal_db" and retrieved:
            top = retrieved[0]
            meta = top.get("metadata", {})
            answer = (
                "Best match: %s (%s) on %s. %s"
                % (
                    meta.get("Name", "Unknown"),
                    meta.get("YearOfRelease", "N/A"),
                    meta.get("Platform", "N/A"),
                    top.get("document", ""),
                )
            )
            return answer + (" [1]" if citations else "")
        if source == "web_search" and web_results:
            top = web_results[0]
            answer = "%s: %s" % (top.get("title", "Result"), top.get("snippet", ""))
            return answer + (" [1]" if citations else "")
        return "I could not find reliable information for that query."

    def _node_synthesize(self, state: Dict[str, Any]):
        if "source" not in state:
            state["source"] = "internal_db"
        if state["source"] == "none" and not state.get("web_results"):
            if self.evaluator.is_sufficient(state["retrieved"]):
                state["source"] = "internal_db"
        citations = self._build_citations(
            source=state["source"],
            retrieved=state["retrieved"],
            web_results=state.get("web_results", []),
        )
        answer = self._synthesize_with_llm(state, citations)
        if not answer:
            answer = self._fallback_answer(state, citations)
        state["answer"] = answer
        state["results"] = (
            state.get("web_results", []) if state["source"] == "web_search" else state["retrieved"]
        )
        state["citations"] = citations
        state["next_node"] = WorkflowStateMachine.END

    def ask(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        state = {
            "query": query,
            "effective_query": query,
            "n_results": n_results,
            "retrieved": [],
            "web_results": [],
            "source": "internal_db",
            "reasoning": "",
            "answer": "",
            "results": [],
            "citations": [],
            "next_node": "rewrite_query",
        }
        final_state = self.workflow.run("rewrite_query", state)
        result = {
            "answer": final_state["answer"],
            "source": final_state["source"],
            "reasoning": final_state["reasoning"],
            "results": final_state["results"],
            "citations": final_state["citations"],
            "effective_query": final_state["effective_query"],
        }
        self.history.append({"query": query, "result": result})
        return result

    def answer_query(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        return self.ask(query=query, n_results=n_results)
