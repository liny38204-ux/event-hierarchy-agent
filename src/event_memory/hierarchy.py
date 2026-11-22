"""Append-only training-free hierarchy for streaming video observations.

The store has three intentionally separate layers:
``segment`` is an observed video interval, ``observation`` is a writer-produced
fact inside that interval, and an ``event`` is a manager decision about one
atomic process.  An ``episode`` is a revisable parent process over events.
Neither manager decision can delete an observation, its timestamp, or evidence.
"""

from __future__ import annotations

from copy import deepcopy
import math
from threading import RLock
from typing import Any


_SEGMENT_FIELDS = {"id", "start", "end", "evidence", "observations"}
_EVIDENCE_FIELDS = {"id", "time", "path"}
_OBSERVATION_FIELDS = {
    "id", "summary", "action_key", "actors", "objects", "state_changes",
    "location", "visible_text", "uncertainty", "evidence_ids",
}


class HierarchyStore:
    """Validated, causal, append-only event and episode directory.

    This is deliberately a memory *representation*, not a learned policy: an
    MLLM may propose a decision, but every reference and update is checked here.
    """

    observation_fields = _OBSERVATION_FIELDS
    event_fields = {"op", "id", "observation_ids", "summary", "action_key", "entities",
                    "continuation_reason", "support_evidence_ids"}

    @classmethod
    def _normalize_observation_content(cls, item):
        cls._text(item["summary"], "observation summary", empty=True)
        cls._text(item["action_key"], "observation action_key", empty=True)
        for key in ("actors", "objects", "state_changes", "visible_text", "uncertainty"):
            item[key] = cls._strings(item[key], key)
        item["location"] = cls._text(item["location"], "location", empty=True)

    def _event_attributes(self, raw, observation_ids):
        action_key = self._text(raw["action_key"], "event action_key")
        canonical = lambda text: " ".join(text.casefold().split())
        if any(canonical(self._observations[oid]["action_key"]) != canonical(action_key)
               for oid in observation_ids):
            raise ValueError("event action_key must match each immutable writer observation; different actions need separate events")
        return {"action_key": action_key, "entities": self._strings(raw["entities"], "event entities")}

    def _validate_continuation_content(self, old, attributes):
        if attributes["action_key"] != old["action_key"]:
            raise ValueError("continuation action_key must match existing event")

    def __init__(self) -> None:
        self._segments: dict[str, dict[str, Any]] = {}
        self._observations: dict[str, dict[str, Any]] = {}
        self._observation_times: dict[str, tuple[float, float]] = {}
        self._evidence: dict[str, dict[str, Any]] = {}
        self._pending: set[str] = set()
        self._events: dict[str, dict[str, Any]] = {}
        self._episodes: dict[str, dict[str, Any]] = {}
        self._event_episode: dict[str, str] = {}
        self._audit_log: list[dict[str, Any]] = []
        self._last_now: float | None = None
        self._last_segment_end: float | None = None
        self._lock = RLock()

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    @staticmethod
    def _text(value: Any, name: str, *, empty: bool = False) -> str:
        if not isinstance(value, str) or (not empty and not value.strip()):
            raise ValueError(f"{name} must be a {'possibly empty ' if empty else 'non-empty '}string")
        return value

    @classmethod
    def _strings(cls, value: Any, name: str) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"{name} must be a list of strings")
        if len(value) != len(set(value)):
            raise ValueError(f"{name} cannot contain duplicates")
        return list(value)

    @classmethod
    def _validate_segment(cls, segment: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
        if not isinstance(segment, dict) or set(segment) != _SEGMENT_FIELDS:
            raise ValueError("segment must contain exactly id, start, end, evidence, observations")
        value = deepcopy(segment)
        cls._text(value["id"], "segment id")
        start, end = cls._finite(value["start"], "segment start"), cls._finite(value["end"], "segment end")
        if start < 0 or start > end:
            raise ValueError("segment requires 0 <= start <= end")
        if not isinstance(value["evidence"], list):
            raise ValueError("segment evidence must be a list")
        evidence_ids: set[str] = set()
        evidence_by_id: dict[str, dict[str, Any]] = {}
        for item in value["evidence"]:
            if not isinstance(item, dict) or set(item) != _EVIDENCE_FIELDS:
                raise ValueError("evidence must contain exactly id, time, path")
            cls._text(item["id"], "evidence id")
            cls._text(item["path"], "evidence path")
            at = cls._finite(item["time"], "evidence time")
            if not start <= at <= end:
                raise ValueError("evidence time must lie within its segment")
            if item["id"] in evidence_ids:
                raise ValueError("duplicate evidence id in segment")
            evidence_ids.add(item["id"])
            evidence_by_id[item["id"]] = deepcopy(item)
        if not isinstance(value["observations"], list):
            raise ValueError("segment observations must be a list")
        observation_ids: set[str] = set()
        observation_times: dict[str, tuple[float, float]] = {}
        normalized_observations = []
        for observation in value["observations"]:
            if not isinstance(observation, dict) or set(observation) != cls.observation_fields:
                raise ValueError("observation must contain exactly the declared writer fields")
            item = deepcopy(observation)
            cls._text(item["id"], "observation id")
            if item["id"] in observation_ids:
                raise ValueError("duplicate observation id in segment")
            observation_ids.add(item["id"])
            cls._normalize_observation_content(item)
            item["evidence_ids"] = cls._strings(item["evidence_ids"], "observation evidence_ids")
            if not item["evidence_ids"]:
                raise ValueError("observation needs at least one evidence id")
            if any(eid not in evidence_by_id for eid in item["evidence_ids"]):
                raise ValueError("observation evidence must belong to its segment")
            times = [evidence_by_id[eid]["time"] for eid in item["evidence_ids"]]
            item["start"], item["end"] = min(times), max(times)
            observation_times[item["id"]] = (item["start"], item["end"])
            normalized_observations.append(item)
        value["start"], value["end"] = start, end
        value["observations"] = normalized_observations
        return value, observation_times

    def add_segment(self, segment: dict[str, Any]) -> None:
        """Append a causally ordered segment and make its facts pending."""
        value, times = self._validate_segment(segment)
        with self._lock:
            if value["id"] in self._segments:
                raise ValueError(f"duplicate segment id: {value['id']}")
            if self._last_segment_end is not None and value["start"] < self._last_segment_end:
                raise ValueError("segments must arrive in non-overlapping causal order")
            identifiers = {item["id"] for item in value["evidence"]}
            if identifiers & self._evidence.keys():
                raise ValueError("evidence ids must be globally unique")
            observation_ids = {item["id"] for item in value["observations"]}
            if observation_ids & self._observations.keys():
                raise ValueError("observation ids must be globally unique")
            self._segments[value["id"]] = deepcopy(value)
            self._evidence.update({item["id"]: deepcopy(item) for item in value["evidence"]})
            self._observations.update({item["id"]: deepcopy(item) for item in value["observations"]})
            self._observation_times.update(times)
            self._pending.update(observation_ids)
            self._last_segment_end = value["end"]

    @staticmethod
    def _require_exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(f"{label} has invalid fields")
        return value

    def _visible(self, now: float) -> None:
        for segment in self._segments.values():
            if segment["end"] > now:
                raise ValueError("future segment exists in manager context")
        for evidence in self._evidence.values():
            if evidence["time"] > now:
                raise ValueError("future evidence exists in manager context")

    @staticmethod
    def _event_bounds(observation_ids: list[str], times: dict[str, tuple[float, float]]) -> tuple[float, float]:
        bounds = [times[oid] for oid in observation_ids]
        return min(x[0] for x in bounds), max(x[1] for x in bounds)

    @staticmethod
    def _episode_bounds(event_ids: list[str], events: dict[str, Any]) -> tuple[float, float]:
        return min(events[eid]["start"] for eid in event_ids), max(events[eid]["end"] for eid in event_ids)

    def apply(self, decision: dict[str, Any], now: float) -> None:
        """Atomically apply event allocations and refreshed episode summaries."""
        now_value = self._finite(now, "now")
        self._require_exact(decision, {"events", "episodes"}, "decision")
        if not isinstance(decision["events"], list) or not isinstance(decision["episodes"], list):
            raise ValueError("events and episodes must be lists")
        with self._lock:
            if self._last_now is not None and now_value < self._last_now:
                raise ValueError("now must not move backwards")
            self._visible(now_value)
            events, episodes = deepcopy(self._events), deepcopy(self._episodes)
            event_episode, pending = deepcopy(self._event_episode), set(self._pending)
            changed_events: set[str] = set()
            used_observations: set[str] = set()
            known_ids = set(self._segments) | set(self._observations) | set(events) | set(episodes)

            for raw in decision["events"]:
                if not isinstance(raw, dict) or raw.get("op") not in {"new", "continue"}:
                    raise ValueError("event must use new or continue")
                kind = raw["op"]
                self._require_exact(raw, self.event_fields, "event")
                event_id = self._text(raw["id"], "event id")
                if event_id in known_ids and not (kind == "continue" and event_id in events):
                    raise ValueError("new event id must be unused")
                if kind == "continue" and event_id not in events:
                    raise ValueError("continue requires an existing event")
                observation_ids = self._strings(raw["observation_ids"], "event observation_ids")
                if not observation_ids:
                    raise ValueError("event needs at least one pending observation")
                if any(oid not in pending for oid in observation_ids):
                    raise ValueError("event may allocate only pending observations")
                if used_observations & set(observation_ids):
                    raise ValueError("observation allocated twice in one decision")
                observation_ids.sort(key=lambda oid: (*self._observation_times[oid], oid))
                summary = self._text(raw["summary"], "event summary")
                attributes = self._event_attributes(raw, observation_ids)
                reason = self._text(raw["continuation_reason"], "continuation_reason", empty=True)
                supports = self._strings(raw["support_evidence_ids"], "support_evidence_ids")
                if not supports or any(eid not in self._evidence for eid in supports):
                    raise ValueError("event support evidence must be known and non-empty")
                if any(self._evidence[eid]["time"] > now_value for eid in supports):
                    raise ValueError("event support references future evidence")
                new_evidence = {eid for oid in observation_ids for eid in self._observations[oid]["evidence_ids"]}
                if not set(supports) & new_evidence:
                    raise ValueError("event support must include new observation evidence")
                if kind == "continue":
                    old = events[event_id]
                    self._validate_continuation_content(old, attributes)
                    if not reason.strip():
                        raise ValueError("continuation_reason is required")
                    old_evidence = set(old["evidence_ids"])
                    if not set(supports) & old_evidence:
                        raise ValueError("continuation support needs evidence from both sides")
                    start = min(old["start"], self._event_bounds(observation_ids, self._observation_times)[0])
                    new_start, end = self._event_bounds(observation_ids, self._observation_times)
                    if new_start < old["end"]:
                        raise ValueError("continued observations must follow the existing event")
                    combined_observations = old["observation_ids"] + observation_ids
                    evidence_ids = list(dict.fromkeys(old["evidence_ids"] + [eid for oid in observation_ids for eid in self._observations[oid]["evidence_ids"]]))
                    history = old["summary_history"] + [summary]
                    version = old["version"] + 1
                else:
                    if reason.strip():
                        raise ValueError("new event cannot claim a continuation reason")
                    start, end = self._event_bounds(observation_ids, self._observation_times)
                    combined_observations = observation_ids
                    evidence_ids = [eid for oid in observation_ids for eid in self._observations[oid]["evidence_ids"]]
                    history, version = [summary], 1
                events[event_id] = {
                    "id": event_id, "summary": summary, "summary_history": history,
                    **attributes,
                    "observation_ids": combined_observations, "evidence_ids": evidence_ids,
                    "start": start, "end": end, "version": version,
                }
                pending.difference_update(observation_ids)
                used_observations.update(observation_ids)
                changed_events.add(event_id)
                known_ids.add(event_id)

            supplied_episode_ids = set()
            episode_members_seen: set[str] = set()
            for raw in decision["episodes"]:
                fields = {"id", "summary", "event_ids", "status"}
                self._require_exact(raw, fields, "episode")
                episode_id = self._text(raw["id"], "episode id")
                if episode_id in supplied_episode_ids:
                    raise ValueError("episode appears twice in a decision")
                supplied_episode_ids.add(episode_id)
                if episode_id in known_ids and episode_id not in episodes:
                    raise ValueError("episode id conflicts with another id")
                summary = self._text(raw["summary"], "episode summary")
                members = self._strings(raw["event_ids"], "episode event_ids")
                if not members or any(eid not in events for eid in members):
                    raise ValueError("episode needs known events")
                # The observer's time, not model list order, defines chronology.
                members.sort(key=lambda eid: (events[eid]["start"],events[eid]["end"],eid))
                if episode_members_seen & set(members):
                    raise ValueError("event has multiple episode parents")
                episode_members_seen.update(members)
                if raw["status"] not in {"open", "closed"}:
                    raise ValueError("episode status must be open or closed")
                old = episodes.get(episode_id)
                if old is not None and not set(old["event_ids"]).issubset(members):
                    raise ValueError("episode updates cannot remove events")
                for event_id in members:
                    prior = event_episode.get(event_id)
                    if prior is not None and prior != episode_id:
                        raise ValueError("events cannot move between episodes")
                start, end = self._episode_bounds(members, events)
                if old is None:
                    version, history = 1, [summary]
                else:
                    version, history = old["version"] + 1, old["summary_history"] + [summary]
                episodes[episode_id] = {"id": episode_id, "summary": summary,
                                        "summary_history": history, "event_ids": members,
                                        "status": raw["status"], "start": start, "end": end,
                                        "version": version}
                for event_id in members:
                    event_episode[event_id] = episode_id
                known_ids.add(episode_id)

            # If an established event changed, its old parent needs a contemporaneous
            # summary update; otherwise manager summaries silently become stale.
            stale = {event_episode[eid] for eid in changed_events if eid in event_episode}
            if not stale.issubset(supplied_episode_ids):
                raise ValueError("updated event requires its parent episode refresh")

            self._events, self._episodes = events, episodes
            self._event_episode, self._pending = event_episode, pending
            self._audit_log.append({"now": now_value, "decision": deepcopy(decision),
                "applied_episode_order": {pid:list(episodes[pid]["event_ids"]) for pid in supplied_episode_ids}})
            self._last_now = now_value

    def context(self) -> dict[str, Any]:
        """Manager-visible causal directory; returned data cannot mutate the store."""
        with self._lock:
            return deepcopy({
                "pending_observation_ids": sorted(self._pending),
                "pending_observations": {oid: self._observations[oid] for oid in sorted(self._pending)},
                "events": self._events,
                "episodes": self._episodes,
                "roots": sorted(set(self._events) - set(self._event_episode)),
                "last_now": self._last_now,
            })

    def snapshot(self) -> dict[str, Any]:
        """Complete deep-copied state for auditing and later retrieval."""
        with self._lock:
            return deepcopy({
                "segments": self._segments,
                "observations": self._observations,
                "events": self._events,
                "episodes": self._episodes,
                "event_episode": self._event_episode,
                "pending_observation_ids": sorted(self._pending),
                "audit_log": self._audit_log,
                "context": self.context(),
            })
