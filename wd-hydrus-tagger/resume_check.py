"""Run-settings line for session.log, so a resume can't continue a batch
with a different model, input file or target.

The first line of a fresh session.log looks like:
    #run: {"input_name": "tags.txt", "input_sha256": "...", "mode": "tags", ...}
"""
import hashlib
import json
import os

HEADER_PREFIX = "#run: "
HEADER_VERSION = 1

# Changing any of these mid-batch changes which files get processed or what
# gets written, so a resume with a different value is refused.
HARD_FIELDS = {
    "model": "model folder",
    "mode": "input mode (--tags or hashes)",
    "input_sha256": "input file contents",
    "ratings_only": "--ratings-only",
    "tag_service": "tag service",
}
# These only change how many tags a file gets, so a difference is reported
# and the resume continues.
SOFT_FIELDS = {
    "input_name": "input file name",
    "threshold": "threshold",
    "max_tags": "max tags",
    "skip_tagged": "--skip-tagged",
}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_settings(model, inputfile, tags_mode, ratings_only, tag_service, threshold, max_tags, skip_tagged):
    return {
        "v": HEADER_VERSION,
        "model": str(model),
        "mode": "tags" if tags_mode else "hashes",
        "input_name": os.path.basename(inputfile),
        "input_sha256": file_sha256(inputfile),
        "ratings_only": bool(ratings_only),
        "tag_service": str(tag_service),
        "threshold": float(threshold),
        "max_tags": int(max_tags),
        "skip_tagged": bool(skip_tagged),
    }


def header_line(settings):
    return HEADER_PREFIX + json.dumps(settings, sort_keys=True)


def parse_header(line):
    """Settings dict from a header line. Raises ValueError if it can't be read."""
    if not line.startswith(HEADER_PREFIX):
        raise ValueError("not a run-settings line")
    data = json.loads(line[len(HEADER_PREFIX):])
    if not isinstance(data, dict):
        raise ValueError("run-settings line is not a JSON object")  # noqa: TRY004 - callers catch ValueError
    return data


def compare(old, new):
    """Return (hard, soft) lists of (label, old_value, new_value) differences."""
    hard, soft = [], []
    for fields, out in ((HARD_FIELDS, hard), (SOFT_FIELDS, soft)):
        for key, label in fields.items():
            if old.get(key) != new.get(key):
                out.append((label, old.get(key), new.get(key)))
    return hard, soft


def mismatch_message(hard, checkpoint_file):
    lines = [f"Resume stopped: the unfinished {checkpoint_file} was started with different settings."]
    for label, old, new in hard:
        if label == "input file contents":
            lines.append(f"  - {label}: the file has changed since that batch started")
        else:
            lines.append(f"  - {label}: was {old!r}, now {new!r}")
    lines.append("Nothing was sent to Hydrus.")
    lines.append("To finish that batch, run the bat file you started it with again.")
    lines.append(f"To start a new batch instead, add --skip-resume (this replaces {checkpoint_file}).")
    return "\n".join(lines)


def soft_message(soft):
    parts = [f"{label} was {old!r}, now {new!r}" for label, old, new in soft]
    return "Note: resuming with changed settings (" + "; ".join(parts) + ")"
