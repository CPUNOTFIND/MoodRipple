"""Helpers for constructing safe QQ private-message UMO routes."""

from __future__ import annotations

from collections.abc import Iterable


def is_direct_friend_origin(origin: str, user_id: str) -> bool:
    """Only reuse a stored route when it is this user's private QQ session."""
    parts = str(origin).strip().split(":", 2)
    return len(parts) == 3 and parts[1].casefold() == "friendmessage" and parts[2] == str(user_id)


def private_origin(user_id: str, stored_origin: str, template: str) -> str:
    if is_direct_friend_origin(stored_origin, user_id):
        return str(stored_origin).strip()
    return (str(template).strip() or "default:FriendMessage:{qq}").format(qq=user_id)


def private_origin_candidates(primary_origin: str, user_id: str, platform_ids: Iterable[str]) -> list[str]:
    candidates = [primary_origin]
    candidates.extend(f"{str(platform_id).strip()}:FriendMessage:{user_id}" for platform_id in platform_ids if str(platform_id).strip())
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))
