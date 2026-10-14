
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from time import time
from typing import Literal
import secrets


# Define the roles allowed in a conversation.
# Literal helps type checkers identify valid role values.
Role = Literal["user", "assistant"]


@dataclass
class Conversation:
    """
    Stores the memory belonging to one conversation.

    Each conversation has its own summary and message history.
    """

    # A compact summary of older conversation turns.
    # It starts empty and will be updated by a separate summarization step.
    summary: str = ""

    # The most recent question-and-answer pairs.
    # These are useful for understanding follow-up questions.
    recent_turns: list[dict[str, str]] = field(default_factory=list)

    # Completed turns that have not yet been included in the summary.
    # We retain them until summarization succeeds.
    pending_summary: list[dict[str, str]] = field(default_factory=list)

    # Used to determine when the conversation was last active.
    updated_at: float = field(default_factory=time)

    # Increments whenever the summary changes.
    # This helps prevent an older summarization result from overwriting
    # a newer result.
    summary_version: int = 0


class ConversationMemory:
    """
    Manages conversational memory for all active conversations.

    This implementation stores data in process memory only.
    Nothing is persisted to a database or file.
    """

    def __init__(
        self,
        max_conversations: int = 500,
        ttl_seconds: int = 3600,
        max_recent_turns: int = 4,
        max_pending_turns: int = 20,
    ):
        # Maximum number of conversations kept in memory.
        self.max_conversations = max_conversations

        # Time-to-live: inactive conversations expire after this many seconds.
        # 3600 seconds = 1 hour.
        self.ttl_seconds = ttl_seconds

        # Number of recent question-and-answer pairs retained in full.
        self.max_recent_turns = max_recent_turns

        # Safety limit for turns waiting to be summarized.
        self.max_pending_turns = max_pending_turns

        # OrderedDict preserves insertion order and lets us evict an older
        # conversation when the conversation limit is reached.
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()

        # Protect shared memory from concurrent access by multiple requests.
        self._lock = RLock()

    @staticmethod
    def new_conversation_id() -> str:
        """
        Generate an unpredictable ID for a new chat.

        The frontend should keep this ID and send it with every question
        belonging to the same conversation.
        """
        return secrets.token_urlsafe(24)

    def _remove_expired(self) -> None:
        """
        Remove conversations that have been inactive for too long.

        This method is called while the memory lock is held.
        """
        now = time()

        expired_ids = [
            conversation_id
            for conversation_id, conversation in self._conversations.items()
            if now - conversation.updated_at > self.ttl_seconds
        ]

        for conversation_id in expired_ids:
            self._conversations.pop(conversation_id, None)

    def _get_or_create(self, conversation_id: str) -> Conversation:
        """
        Return an existing conversation or create a new one.

        This internal method must be called while holding self._lock.
        """
        self._remove_expired()

        conversation = self._conversations.get(conversation_id)

        if conversation is None:
            # Evict the oldest stored conversation if capacity is reached.
            while len(self._conversations) >= self.max_conversations:
                self._conversations.popitem(last=False)

            conversation = Conversation()
            self._conversations[conversation_id] = conversation

        else:
            # Mark this conversation as recently used in the ordered mapping.
            self._conversations.move_to_end(conversation_id)

        # Update activity time whenever the conversation is accessed.
        conversation.updated_at = time()

        return conversation

    def get_context(self, conversation_id: str) -> dict:
        """
        Return the existing summary and recent turns for a conversation.

        The caller can use this context to decide whether a question
        needs rewriting before document retrieval.
        """
        with self._lock:
            conversation = self._get_or_create(conversation_id)

            # Return copies so callers cannot accidentally modify stored lists.
            return {
                "summary": conversation.summary,
                "recent_turns": [
                    turn.copy() for turn in conversation.recent_turns
                ],
            }

    def add_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
    ) -> None:
        """
        Store one completed question-and-answer pair.

        Call this only after the answer has been generated successfully.
        """
        with self._lock:
            conversation = self._get_or_create(conversation_id)

            # Do not silently discard unsummarized history.
            # The caller should summarize pending turns before reaching
            # this limit, then retry adding the turn if necessary.
            if len(conversation.pending_summary) >= self.max_pending_turns:
                raise RuntimeError(
                    "Too many unsummarized turns. "
                    "Summarize pending history before adding another turn."
                )

            # Store the completed exchange as one turn.
            turn = {
                "question": question,
                "answer": answer,
            }

            # Keep the exchange in recent history.
            conversation.recent_turns.append(turn.copy())

            # Also track it until it has been summarized.
            conversation.pending_summary.append(turn.copy())

            # Trim only the recent-history window.
            # The pending list is retained so older turns are not lost.
            if len(conversation.recent_turns) > self.max_recent_turns:
                conversation.recent_turns = conversation.recent_turns[
                    -self.max_recent_turns:
                ]

            conversation.updated_at = time()

    def get_pending_summary(self, conversation_id: str) -> dict:
        """
        Return older unsummarized turns that are ready for summarization.

        Recent turns stay available in full, so they do not need to be
        included in the older-turn summarization batch.

        The returned version is used to detect stale summary updates.
        """
        with self._lock:
            conversation = self._get_or_create(conversation_id)

            # Leave the latest turns in the recent-history window.
            # Only older pending turns are selected for summarization.
            if len(conversation.pending_summary) <= self.max_recent_turns:
                turns_to_summarize = []
            else:
                turns_to_summarize = conversation.pending_summary[
                    : -self.max_recent_turns
                ]

            return {
                # The summarizer should incorporate this existing summary
                # rather than replacing it with information from only new turns.
                "summary": conversation.summary,

                # Return copies so external code cannot mutate stored history.
                "turns": [
                    turn.copy() for turn in turns_to_summarize
                ],

                # Remember which summary version was read.
                "summary_version": conversation.summary_version,

                # Remember how many turns were included in this snapshot.
                "turn_count": len(turns_to_summarize),
            }

    def commit_summary(
        self,
        conversation_id: str,
        summary: str,
        summarized_turns: int,
        expected_version: int,
    ) -> bool:
        """
        Save a newly generated summary and mark its turns as summarized.

        Returns True if the update was committed.
        Returns False if the conversation disappeared or the summary
        changed while summarization was running.
        """
        with self._lock:
            # Do not create a new conversation just to commit an old result.
            conversation = self._conversations.get(conversation_id)

            if conversation is None:
                return False

            # Prevent an older summarization operation from overwriting
            # a summary committed by another request.
            if conversation.summary_version != expected_version:
                return False

            # Validate the number of turns being removed from pending history.
            if not 0 <= summarized_turns <= len(conversation.pending_summary):
                return False

            # Store the newly generated rolling summary.
            conversation.summary = summary

            # Remove only the turns included in the successful summary.
            # Any turns appended after the snapshot remain pending.
            del conversation.pending_summary[:summarized_turns]

            # Mark that the summary has changed.
            conversation.summary_version += 1
            conversation.updated_at = time()

            return True

    def clear(self, conversation_id: str) -> bool:
        """
        Delete a conversation's memory.

        Returns True if a conversation was removed, otherwise False.
        """
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None


# Create one memory manager for this running Python process.
# Other modules can import this instance instead of creating separate managers.
conversation_memory = ConversationMemory()
