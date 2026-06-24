"""
Chat message history with context-window-aware trimming.

A ChatHistory holds an ordered list of {role, content} messages and can produce
a payload trimmed to fit a character budget derived from the context window.
The system prompt is supplied separately at build time so it is never trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str        # "user" | "assistant" | "system"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatHistory:
    messages: list[Message] = field(default_factory=list)
    max_turns: int = 200          # hard cap on stored messages

    # ── Mutators ──────────────────────────────────────────────────────────────

    def add(self, role: str, content: str) -> Message:
        msg = Message(role, content)
        self.messages.append(msg)
        if len(self.messages) > self.max_turns:
            # Drop oldest, keeping pairs roughly intact
            overflow = len(self.messages) - self.max_turns
            self.messages = self.messages[overflow:]
        return msg

    def add_user(self, content: str) -> Message:
        return self.add("user", content)

    def add_assistant(self, content: str) -> Message:
        return self.add("assistant", content)

    def clear(self) -> None:
        self.messages.clear()

    # ── Accessors ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.messages)

    def is_empty(self) -> bool:
        return not self.messages

    def to_list(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]

    # ── Payload construction ─────────────────────────────────────────────────

    def build_payload(
        self,
        system_prompt: str = "",
        num_ctx: int = 4096,
        budget_fraction: float = 0.5,
        chars_per_token: int = 4,
    ) -> list[dict]:
        """
        Build a messages payload trimmed to fit the context budget.

        Budget (in characters) ≈ num_ctx * budget_fraction * chars_per_token.
        The system prompt is always included and counted against the budget.
        The most recent messages are preferred; oldest are dropped first.
        Always keeps at least the final 2 messages.
        """
        budget_chars = max(0, int(num_ctx * budget_fraction * chars_per_token) - len(system_prompt))

        history = list(self.messages)
        total = sum(len(m.content) for m in history)
        while total > budget_chars and len(history) > 2:
            removed = history.pop(0)
            total -= len(removed.content)

        payload: list[dict] = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        payload.extend(m.to_dict() for m in history)
        return payload

    # ── Transcript ─────────────────────────────────────────────────────────────

    def transcript(self) -> str:
        """Plain-text transcript for copy/export."""
        label = {"user": "You", "assistant": "Assistant", "system": "System"}
        out = []
        for m in self.messages:
            out.append(f"{label.get(m.role, m.role)}:\n{m.content.strip()}\n")
        return "\n".join(out)
