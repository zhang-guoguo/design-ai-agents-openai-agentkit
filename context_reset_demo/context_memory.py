from typing import List, Dict, Optional
import json

class ContextMemory:
    """
    Two-tier memory:
      - profile_facts: persistent traits (name, preferences) that survive topic changes
      - thread: recent turns + summary (cleared on new topic)
    """

    def _init_(self, max_recent_pairs: int = 4):
        self.max_recent_pairs = max_recent_pairs
        self.profile_facts: List[str] = []
        self.thread_recent: List[Dict[str, str]] = []
        self.thread_summary: Optional[str] = None
        self ._ pending_overflow: List[Dict[str, str]] = []


    def add_user(self, text: str):
        self.thread_recent.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.thread_recent.append({"role": "assistant", "content": text})
        self ._ trim()


    def _trim(self):
        max_msgs = self.max_recent_pairs * 2
        if len(self.thread_recent) > max_msgs:
            overflow = self.thread_recent [ :- max_msgs]
            self.thread_recent = self.thread_recent [-max_msgs : ]
            self._ pending_overflow = overflow
        else:
            self._ pending_overflow = []

    def pop_pending_overflow(self) -> List[Dict[str, str]]:
        ov = self ._ pending_overflow
        self ._ pending_overflow = []
        return ov

    def context_messages(self) -> List[Dict[str, str]]:
        ctx: List[Dict[str, str]] = []
        if self.profile_facts:
            facts = "\n".join(f"- {f}" for f in self.profile_facts)
            ctx.append({"role": "system", "content": f"Persistent user/profile facts: \n{facts}"})
        if self.thread_summary:
            ctx.append({"role": "system", "content": f"Thread summary: \n{self.thread_summary}"})
        ctx.extend(self.thread_recent)
        return ctx

    def append_summary(self, chunk: str):
        if not chunk:
            return
        if self.thread_summary:
            self.thread_summary += "\n" + chunk
        else:
            self.thread_summary = chunk

    def upsert_profile_facts(self, new_facts: List[str], max_total: int = 32):
        def norm(s: str) -> str:
            return s.strip().rstrip(".")
        existing = {norm(x) for x in self.profile_facts}
        for f in new_facts:
            k = norm(f)
            if k and k not in existing:
                self.profile_facts. append(f.strip())
                existing. add(k)
        if len(self.profile_facts) > max_total:
            self.profile_facts = self.profile_facts[-max_total:]

    def new_topic(self):
        """Soft reset: keep profile facts, clear thread memory."""
        self.thread_recent = []
        self.thread_summary = None
        self ._ pending_overflow = []

    def reset_all(self):
        """Hard reset: clear everything. """
        self.profile_facts = []
        self.new_topic()

    def get_state(self) -> Dict:
        return {
            "max_recent_pairs": self.max_recent_pairs,
            "profile_facts": self.profile_facts,
            "thread_recent": self.thread_recent,
            "thread_summary": self.thread_summary
        }

    def load_state(self, state: Dict):
        if not state: 
            return
        self.max_recent_pairs = state.get("max_recent_pairs", self.max_recent_pairs)
        self.profile_facts = state.get("profile_facts", [])
        self.thread_recent = state.get("thread_recent", [])
        self.thread_summary = state.get("thread_summary")