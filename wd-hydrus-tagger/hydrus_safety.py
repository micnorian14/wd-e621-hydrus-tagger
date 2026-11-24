"""Checks that run before anything is written to Hydrus.

Everything here is plain Python with no Hydrus calls, so it can be tested
without a running client.
"""
import re

import requests

# Hydrus service types (hydrus/core/HydrusConstants.py)
SERVICE_TYPE_TAG_REPOSITORY = 0
SERVICE_TYPE_LOCAL_TAG = 5
SERVICE_TYPE_COMBINED_TAG = 10

# Key of the built-in "my tags" service (b"local tags".hex())
MY_TAGS_SERVICE_KEY = "6c6f63616c2074616773"

DEFAULT_MAX_TAGS = 150
DEFAULT_TIMEOUT = (10, 600)  # (connect, read) seconds; big searches can take minutes

_HEX64 = re.compile(r"[0-9a-f]{64}")


class SafetyError(Exception):
    """Raised when a safety check fails and nothing should be written."""


def normalize_hash(value):
    """Return a lowercase 64-hex sha256 string, or None if it isn't one.

    A leading "sha256:" (as Hydrus copies it) is removed first.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.lower().startswith("sha256:"):
        value = value[7:].strip()
    value = value.lower()
    return value if _HEX64.fullmatch(value) else None


def _service_entries(services_json):
    """List of (key, name, type) from a /get_services response.

    Prefers "services_v2" (a list). Falls back to the older "services" dict.
    """
    if not isinstance(services_json, dict):
        raise SafetyError("Unexpected /get_services response from Hydrus")
    entries = []
    v2 = services_json.get("services_v2")
    if isinstance(v2, list):
        for s in v2:
            if isinstance(s, dict):
                entries.append((str(s.get("service_key", "")).lower(), s.get("name"), s.get("type")))
        return entries
    legacy = services_json.get("services")
    if isinstance(legacy, dict):
        for key, s in legacy.items():
            if isinstance(s, dict):
                entries.append((str(key).lower(), s.get("name"), s.get("type")))
        return entries
    raise SafetyError("Hydrus /get_services response has no service list")


def resolve_tag_service(services_json, name=None, key=None, allow_my_tags=False):
    """Find the one local tag service to write to. Returns (key, name, type).

    Matches by key if one is given, otherwise by exact name. Refuses anything
    that is not exactly one local tag service (type 5).
    """
    entries = _service_entries(services_json)
    if key:
        wanted = str(key).strip().lower()
        matches = [e for e in entries if e[0] == wanted]
        label = f"key '{wanted}'"
    elif name:
        matches = [e for e in entries if e[1] == name]
        label = f"name '{name}'"
    else:
        raise SafetyError("No tag service set. Set 'tag_service' (or 'tag_service_key') in config.yaml or pass --tag-service")

    local_tag = [f"'{e[1]}' ({e[0]})" for e in entries if e[2] == SERVICE_TYPE_LOCAL_TAG]
    available = ", ".join(local_tag) if local_tag else "none"

    if not matches:
        raise SafetyError(f"No Hydrus service with {label}. Local tag services: {available}")
    if len(matches) > 1:
        keys = ", ".join(e[0] for e in matches)
        raise SafetyError(f"More than one Hydrus service has {label} ({keys}). Set 'tag_service_key' in config.yaml to pick one")

    s_key, s_name, s_type = matches[0]
    if s_type == SERVICE_TYPE_TAG_REPOSITORY:
        raise SafetyError(f"'{s_name}' is a tag repository (like the PTR). Refusing to write there; use a local tag service")
    if s_type == SERVICE_TYPE_COMBINED_TAG:
        raise SafetyError(f"'{s_name}' is the 'all known tags' view, not a real service. Use a local tag service")
    if s_type != SERVICE_TYPE_LOCAL_TAG:
        raise SafetyError(f"'{s_name}' is not a tag service (type {s_type}). Use a local tag service")
    if (s_key == MY_TAGS_SERVICE_KEY or str(s_name).strip().lower() == "my tags") and not allow_my_tags:
        raise SafetyError(f"'{s_name}' is your 'my tags' service. Refusing unless you pass --allow-my-tags")
    return s_key, s_name, s_type


def sanitize_tags(tags, keep_namespace=False):
    """Clean a tag list right before it is sent.

    Keeps strings only, strips whitespace, drops empties and duplicates (order
    kept). Unless keep_namespace is set, a tag that contains ':' but doesn't
    start with one gets a ':' in front, so Hydrus stores it whole instead of
    splitting it into a namespace (e.g. 'rem_(re:zero)' -> ':rem_(re:zero)').
    Use keep_namespace=True only for tags this tool builds on purpose, such as
    'rating:general'.
    """
    out = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag = tag.strip()
        if not tag:
            continue
        if not keep_namespace and ":" in tag and not tag.startswith(":"):
            tag = ":" + tag
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def select_tags(tag_scores, threshold, max_tags):
    """Tags scoring above threshold, highest first, at most max_tags of them."""
    picked = [(t, s) for t, s in tag_scores.items() if s > threshold]
    picked.sort(key=lambda ts: ts[1], reverse=True)
    if max_tags is not None and max_tags > 0:
        picked = picked[:max_tags]
    return [t for t, _ in picked]


def check_threshold(threshold):
    """Raise if threshold is outside 0..1; return a warning string or None."""
    try:
        t = float(threshold)
    except (TypeError, ValueError):
        raise SafetyError(f"Threshold must be a number, got {threshold!r}") from None
    if not 0 < t < 1:
        raise SafetyError(f"Threshold must be between 0 and 1, got {t}")
    if t < 0.1 or t > 0.99:
        return f"Warning: threshold {t} is unusual (normal range is 0.1 to 0.99)"
    return None


def check_max_tags(max_tags):
    try:
        m = int(max_tags)
    except (TypeError, ValueError):
        raise SafetyError(f"max_tags must be a whole number, got {max_tags!r}") from None
    if m < 1:
        raise SafetyError(f"max_tags must be at least 1, got {m}")
    return m


def metadata_row_for_hash(metadata_json, file_hash):
    """Return the metadata row for exactly this hash if Hydrus has the file.

    Returns (row, None) on success or (None, reason) if the row is missing,
    the hash doesn't match, or file_id is null (Hydrus doesn't have the file).
    """
    rows = metadata_json.get("metadata") if isinstance(metadata_json, dict) else metadata_json
    if not isinstance(rows, list) or not rows:
        return None, "no metadata returned"
    for row in rows:
        if isinstance(row, dict) and str(row.get("hash", "")).lower() == file_hash:
            if row.get("file_id") is None:
                return None, "Hydrus does not have this file (no file_id)"
            return row, None
    return None, "metadata hash does not match the requested hash"


class TimeoutSession(requests.Session):
    """requests.Session that applies a default timeout to every request."""

    def __init__(self, timeout=DEFAULT_TIMEOUT):
        super().__init__()
        self._default_timeout = timeout

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(method, url, **kwargs)
