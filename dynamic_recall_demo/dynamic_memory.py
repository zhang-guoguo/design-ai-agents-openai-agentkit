from typing import List, Dict, Optional
import json

class DynamicMemory:
    """
    Rolling conversation memory that:
    - stores recent messages (short buffer)
    - keeps a rolling summary string of older turns
    - maintains a small set of extracted facts
    - persists to disk (via get_state/load_state)
    """

    def __init__(self, max_recent_pairs: int = 2, auto_summarize: bool = True):
        self.max_recent_pairs = max_recent_pairs
        self.auto_summarize = auto_summarize
        self.recent: List[Dict[str, str]] = []
        self.summary: Optional[str] = None
        self.facts: List[str] = []

    def add_user(self, text: str):
        self.recent.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.recent.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        max_msgs = self.max_recent_pairs * 2
        if len(self.recent) > max_msgs and self.auto_summarize:
            overflow = self.recent[:-max_msgs]
            self.recent = self.recent[-max_msgs:]
            self._pending_overflow = overflow

    def pop_pending_overflow(self) -> List[Dict[str, str]]:
        ov = getattr(self, "_pending_overflow", [])
        self._pending_overflow = []
        return ov

    def context_messages(self) -> List[Dict[str, str]]:
        ctx: List[Dict[str, str]] = []
        if self.summary:
            ctx.append(
                {
                    "role": "system",
                    "content": f"Rolling summary of earlier conversation:\n{self.summary}",
                }
            )
        if self.facts:
            joined = "\n".join(f"- {f}" for f in self.facts)
            ctx.append(
                {
                    "role": "system",
                    "content": f"Known facts extracted from earlier turns:\n{joined}",
                }
            )
        ctx.extend(self.recent)
        return ctx

    def append_to_summary(self, new_summary_chunk: str):
        if not new_summary_chunk:
            return
        if self.summary:
            self.summary += "\n" + new_summary_chunk
        else:
            self.summary = new_summary_chunk

    def upsert_facts(self, new_facts: List[str], max_total: int = 24):
        norm = lambda s: s.strip().rstrip(".")
        existing = {norm(x) for x in self.facts}
        for nf in new_facts:
            key = norm(nf)
            if key and key not in existing:
                self.facts.append(nf.strip())
                existing.add(key)
        if len(self.facts) > max_total:
            self.facts = self.facts[-max_total:]

    def get_state(self) -> Dict:
        return {
            "max_recent_pairs": self.max_recent_pairs,
            "auto_summarize": self.auto_summarize,
            "summary": self.summary,
            "facts": self.facts,
            "recent": self.recent,
        }

    def load_state(self, state: Dict):
        if not state:
            return
        self.max_recent_pairs = state.get("max_recent_pairs", self.max_recent_pairs)
        self.auto_summarize = state.get("auto_summarize", self.auto_summarize)
        self.summary = state.get("summary")
        self.facts = state.get("facts", [])
        self.recent = state.get("recent", [])
