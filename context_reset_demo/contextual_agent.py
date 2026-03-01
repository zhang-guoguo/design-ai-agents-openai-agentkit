import os, sys, json
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from context_memory import ContextMemory

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STATE_PATH = "context_state.json"
SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant. Use provided profile facts and thread summary to stay consistent. "
    "Be concise and precise."
)

def load_memory() -> ContextMemory:
    mem = ContextMemory(max_recent_pairs=4)
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                mem. load_state(json.load(f))
        except Exception:
          pass
    return mem

def save_memory(mem: ContextMemory) :
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(mem.get_state(), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _chat(messages, model="gpt-4.1-mini", temperature=0.3) -> str:
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
        return resp.choices[0].message.content.strip()
    return ""

def _extract_profile_facts(text: str) -> List[str]:
    sys_prompt = (
        "Extract durable user/profile facts from the text: name, preferences, timezone, tools, constraints. "
        "One fact per line, no numbering, keep short."
    )
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}
    ]
    facts_text = _chat(msgs, model="gpt-4.1-mini", temperature=0.0)
    return [ln.strip(" -. ").strip() for ln in facts_text.splitlines() if ln.strip()]

def _summarize_overflow(mem: ContextMemory):
    overflow = mem.pop_pending_overflow()
    if not overflow:
      return
    transcript = "\n".join([f"{m['role' ]. upper()}: {m['content']}" for m in overflow])
    sum_msgs = [
        {"role": "system", "content": "Summarize into 3-6 concise bullets with facts, decisions, deadlines."},
        {"role": "user", "content": transcript}
    ]
    chunk = _chat(sum_msgs, model="gpt-4.1-mini", temperature=0.2)
    if chunk:
        mem. append_summary(chunk)

    facts = _extract_profile_facts(transcript)
    if facts:
        mem.upsert_profile_facts(facts)

def chat_once(user_text: str, mem: ContextMemory) -> str:
    mem. add_user(user_text)

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS} ]
    messages += mem. context_messages ()
    messages. append({"role": "user", "content": user_text})

    reply = _chat(messages, model="gpt-4.1-mini", temperature=0.3) or "(No reply)"
    mem.add_assistant(reply)

    if len(mem.thread_recent) >= 2:
        last_two = mem. thread_recent[-2:]
        mini = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in last_two])
        facts = extract_profile_facts(mini)
        if facts:
          mem. upsert_profile_facts(facts)

    _summarize_overflow(mem)
    return reply

USAGE = """
Usage:
    python contextual_agent.py "Your message here"  # continue current thread
    python contextual_agent.py -- new-topic         # soft reset: keep profile facts, clear thread
    python contextual_agent.py -- reset-all         # hard reset: clear everything
    python contextual_agent.py -- show              # inspect memory
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE); sys.exit(0)

    mem = load_memory ()

    if sys.argv[1] == " -- reset-all":
        mem.reset_all(); save_memory(mem)
        print("All memory cleared (profile facts + thread)."); sys.exit(0)

    if sys.argv[1] == " -- new-topic":
        mem.new_topic(); save_memory(mem)
        print("New topic started (kept profile facts, cleared thread)."); sys.exit(0)

    if sys.argv[1] == " -- show":
        print(" ----- PROFILE FACTS -----")
        if mem.profile_facts:
            for f in mem.profile_facts: print("-", f)
        else:
            print("(none)")
        print("\n ----- THREAD SUMMARY -----")
        print(mem.thread_summary or "(none)")
        print("\n ----- THREAD RECENT -----")
        for m in mem.thread_recent:
            who = "You" if m["role"] == "user" else "Assistant"
            print(f"{who}: {m['content' ]}")
        sys.exit(0)

    msg = " ".join(sys.argv[1:])
    reply = chat_once(msg, mem)
    save_memory(mem)
    print("Assistant:", reply)