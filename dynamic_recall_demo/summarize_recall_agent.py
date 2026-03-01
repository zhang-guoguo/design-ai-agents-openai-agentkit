import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from dynamic_memory import DynamicMemory


load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://codex-api-slb.packycode.com/v1"
)

STATE_PATH = "dynamic_state.json"

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant. Use provided summary/facts to maintain continuity. "
    "Be concise and precise."
)


def load_memory() -> DynamicMemory:
    mem = DynamicMemory(max_recent_pairs=2, auto_summarize=True)
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                mem.load_state(json.load(f))
        except Exception:
            pass
    return mem


def save_memory(mem: DynamicMemory):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(mem.get_state(), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _model_chat(messages, model="gpt-5.2-2025-12-11", temperature=0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
        return resp.choices[0].message.content.strip()
    return ""


def _extract_facts_from_text(text: str) -> list[str]:
    """Use model to extract factual bullet points."""
    messages = [
        {
            "role": "system",
            "content": "Extract key facts, deadlines, or preferences as bullet points. One fact per line.",
        },
        {"role": "user", "content": text},
    ]
    facts_text = _model_chat(messages, model="gpt-5.2-2025-12-11", temperature=0.0)
    return [ln.strip("-• ").strip() for ln in facts_text.splitlines() if ln.strip()]


def summarize_overflow(mem: DynamicMemory):
    """Summarize and extract facts when older messages overflow."""
    overflow = mem.pop_pending_overflow()
    if not overflow:
        return

    transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in overflow])

    sum_messages = [
        {"role": "system", "content": "Summarize this chat briefly into key points and facts."},
        {"role": "user", "content": transcript},
    ]
    summary_chunk = _model_chat(sum_messages)
    if summary_chunk:
        mem.append_to_summary(summary_chunk)


    new_facts = _extract_facts_from_text(transcript)
    if new_facts:
        mem.upsert_facts(new_facts)


def _ensure_facts_from_latest_pair(mem: DynamicMemory):
    """Always extract facts from the latest user-assistant pair."""
    if len(mem.recent) < 2:
        return
    last_two = mem.recent[-2:]
    transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in last_two])
    new_facts = _extract_facts_from_text(transcript)
    if new_facts:
        mem.upsert_facts(new_facts)


def chat_with_memory(user_text: str, mem: DynamicMemory) -> str:
    """Regular conversation with auto fact-extraction and summarization."""
    mem.add_user(user_text)
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    messages += mem.context_messages()
    reply = _model_chat(messages, model="gpt-5.2-2025-12-11", temperature=0.3)
    if not reply:
        reply = "(No reply)"
    mem.add_assistant(reply)

    _ensure_facts_from_latest_pair(mem)
    summarize_overflow(mem)
    return reply


def recall_question(question: str, mem: DynamicMemory) -> str:
    """Answer recall questions from summary + facts, fallback to recent buffer."""
    sum_text = mem.summary or "(no summary yet)"
    facts_text = "\n".join(f"- {f}" for f in mem.facts) if mem.facts else "(no facts yet)"

    base_messages = [
        {
            "role": "system",
            "content": "Answer strictly using the provided memory content. If unsure, say you don't know.",
        },
        {"role": "system", "content": f"Summary:\n{sum_text}"},
        {"role": "system", "content": f"Facts:\n{facts_text}"},
    ]

    if not mem.facts and mem.recent:
        recent_blob = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in mem.recent])
        base_messages.append({"role": "system", "content": f"Recent messages:\n{recent_blob}"})

    base_messages.append({"role": "user", "content": question})
    answer = _model_chat(base_messages, model="gpt-5.2-2025-12-11", temperature=0.0)
    return answer or "(No recall answer)"


USAGE = """
Usage:
    python summarize_recall_agent.py "Your message here"
    python summarize_recall_agent.py "recall: Your recall question"
    python summarize_recall_agent.py --show
    python summarize_recall_agent.py --reset
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)

    if sys.argv[1] == "--reset":
        if os.path.exists(STATE_PATH):
            os.remove(STATE_PATH)
        print(" ✅ Memory cleared. ")
        sys.exit(0)

    mem = load_memory()

    if sys.argv[1] == "--show":
        print("----- SUMMARY -----")
        print(mem.summary or "(no summary)")
        print("\n----- FACTS -----")
        if mem.facts:
            for f in mem.facts:
                print("-", f)
        else:
            print("(no facts)")
        print("\n----- RECENT -----")
        for m in mem.recent:
            who = "You" if m["role"] == "user" else "Assistant"
            print(f"{who}: {m['content']}")
        sys.exit(0)

    arg = " ".join(sys.argv[1:])
    if arg.lower().startswith("recall:"):
        question = arg.split(":", 1)[1].strip()
        answer = recall_question(question, mem)
        print("Assistant:", answer)
        save_memory(mem)
        sys.exit(0)

    user_text = arg
    reply = chat_with_memory(user_text, mem)
    save_memory(mem)
    print("Assistant:", reply)
