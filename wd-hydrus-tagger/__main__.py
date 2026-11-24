import json
import os.path
import sys
from io import BytesIO

import click
import hydrus_api
import imageio.v3 as iio
import numpy as np
import yaml
from PIL import Image, ImageFile

from . import hydrus_safety, interrogate, resume_check
from .hydrus_safety import SafetyError

Image.MAX_IMAGE_PIXELS = None

DEFAULT_MODEL = "wd-v1-4-vit-tagger-v2"
DEFAULT_THRESHOLD = 0.35
DEFAULT_HOST = "http://127.0.0.1:45869"
DEFAULT_TAG_SERVICE = "A.I. Tags"


def load_config(path="config.yaml"):
    """Load optional config.yaml from the current working directory."""
    if os.path.isfile(path):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            click.echo(f"Warning: could not load {path}: {e}", err=True)
    return {}


def resolve(cli_value, config_key, fallback, config):
    """Prefer an explicitly supplied CLI value; otherwise use config.yaml, then the hard-coded fallback."""
    if cli_value is not None:
        return cli_value
    value = config.get(config_key)
    return fallback if value is None else value


def to_bool(value):
    """Read a true/false setting that may come from YAML as a bool, number or word."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(value)


# Older bat files and docs pass these as "--cpu 0", "--privacy true", etc.
# They are on/off flags now, so translate the old form before click sees it.
_LEGACY_BOOL_FLAGS = {
    "--cpu": ("--cpu", "--gpu"),
    "--privacy": ("--privacy", "--show-tags"),
    "--ratings-only": ("--ratings-only", None),
}
_TRUE_WORDS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_WORDS = {"0", "false", "f", "no", "n", "off"}


def upgrade_legacy_bool_args(argv):
    out = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        name, eq, value = arg.partition("=")
        if name in _LEGACY_BOOL_FLAGS:
            on_flag, off_flag = _LEGACY_BOOL_FLAGS[name]
            word = None
            if eq:
                word = value.strip().lower()
            elif i + 1 < len(argv) and argv[i + 1].strip().lower() in _TRUE_WORDS | _FALSE_WORDS:
                word = argv[i + 1].strip().lower()
                i += 1
            if word in _TRUE_WORDS:
                out.append(on_flag)
            elif word in _FALSE_WORDS:
                if off_flag:
                    out.append(off_flag)
            else:
                out.append(arg)
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
    ":d",
    ":3",
    ":o",
    ";d",
    ":>="
]


def debug(msg):
    """Print a debug line and record it for session.log."""
    click.echo(msg)
    interrogate.session_debug.append(msg)


def load_model(model, cpu):
    """Read model/<model>/info.json and load the model. Returns (modelinfo, interrogator)."""
    if not os.path.isfile('./model/' + model + '/info.json'):
        raise click.ClickException(f"info.json not found in model folder 'model/{model}'!")

    with open('./model/' + model + '/info.json') as json_f:
        modelinfo = json.load(json_f)

    integerator = interrogate.WaifuDiffusionInterrogator(
        modelinfo['modelname'],
        modelinfo['modelfile'],
        modelinfo['tagsfile'],
        model,
        modelinfo['ratingsflag'],
        modelinfo['numberofratings'],
        repo_id=modelinfo['source'],
    )
    integerator.load(cpu)
    return modelinfo, integerator


def common_settings(config, cpu, model, threshold, max_tags):
    """Settings shared by every command: CLI flag, then config.yaml, then built-in default."""
    settings = {
        "cpu": to_bool(resolve(cpu, "cpu", False, config)),
        "model": resolve(model, "model", DEFAULT_MODEL, config),
        "threshold": resolve(threshold, "threshold", DEFAULT_THRESHOLD, config),
        "max_tags": resolve(max_tags, "max_tags", hydrus_safety.DEFAULT_MAX_TAGS, config),
    }
    try:
        warning = hydrus_safety.check_threshold(settings["threshold"])
        settings["threshold"] = float(settings["threshold"])
        settings["max_tags"] = hydrus_safety.check_max_tags(settings["max_tags"])
    except SafetyError as e:
        raise click.UsageError(str(e)) from None
    if warning:
        click.echo(warning, err=True)
    return settings


def api_settings(config, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy):
    """Settings for the Hydrus commands."""
    settings = common_settings(config, cpu, model, threshold, max_tags)
    settings["token"] = resolve(token, "token", None, config)
    if not settings["token"]:
        raise click.UsageError("API token required. Set 'token' in config.yaml or pass --token")
    settings["host"] = resolve(host, "host", DEFAULT_HOST, config)
    # An explicit --tag-service wins. Otherwise a pinned tag_service_key in
    # config.yaml wins over the tag_service name.
    if tag_service is not None:
        settings["tag_service"], settings["tag_service_key"] = tag_service, None
    else:
        settings["tag_service"] = resolve(None, "tag_service", DEFAULT_TAG_SERVICE, config)
        settings["tag_service_key"] = config.get("tag_service_key") or None
    settings["ratings_only"] = to_bool(resolve(ratings_only, "ratings_only", False, config))
    settings["privacy"] = to_bool(resolve(privacy, "privacy", True, config))
    settings["timeout"] = (10, resolve(None, "request_timeout", hydrus_safety.DEFAULT_TIMEOUT[1], config))
    return settings


def connect(settings, allow_my_tags):
    """Connect to Hydrus and find the one local tag service to write to."""
    session = hydrus_safety.TimeoutSession(settings["timeout"])
    client = hydrus_api.Client(settings["token"], settings["host"], session=session)
    services = client.get_services()
    try:
        service = hydrus_safety.resolve_tag_service(
            services,
            name=settings["tag_service"],
            key=settings["tag_service_key"],
            allow_my_tags=allow_my_tags,
        )
    except SafetyError as e:
        raise click.ClickException(str(e)) from None
    debug(f"DEBUG: target tag service = '{service[1]}' (key {service[0]}, type {service[2]})")
    return client, service


def pick_rating(modelinfo, ratings):
    """Highest-scoring rating name, or None if the model has no ratings."""
    if not modelinfo['ratingsflag'] or not ratings:
        return None
    return max(ratings, key=lambda k: ratings[k])


def build_tags(modelinfo, ratings, tag_scores, threshold, max_tags, ratings_only):
    """Return (rating, content_tags, final_tags). final_tags is exactly what gets sent."""
    rating = pick_rating(modelinfo, ratings)
    content = [] if ratings_only else hydrus_safety.select_tags(tag_scores, threshold, max_tags)
    content = hydrus_safety.sanitize_tags(content)
    own = []
    if rating:
        own.append("rating:" + rating)
    if ratings_only:
        own.append("ratings only " + modelinfo['modelname'] + " ai generated tags")
    else:
        own.append(modelinfo['modelname'] + " ai generated tags")
    # Our own tags keep their namespace (rating:...), everything is deduped.
    final = hydrus_safety.sanitize_tags(content + hydrus_safety.sanitize_tags(own, keep_namespace=True), keep_namespace=True)
    return rating, content, final


def describe_send(file_hash, service, final_tags, show_tags=True):
    text = f"{file_hash} -> '{service[1]}' ({service[0]}, type {service[2]}): {len(final_tags)} tags"
    if show_tags:
        text += ": " + ", ".join(final_tags)
    return text


@click.group()
def cli():
    pass


@click.command()
@click.argument("filename")
@click.option("--cpu/--gpu", default=None, help="Run the model on the CPU or the GPU (default: config.yaml, else GPU)")
@click.option("--model", default=None, help=f"The tagging model folder to use (default: config.yaml, else {DEFAULT_MODEL})")
@click.option("--threshold", type=float, default=None, help=f"The threshold to drop tags below (default: config.yaml, else {DEFAULT_THRESHOLD})")
@click.option("--max-tags", type=int, default=None, help=f"Keep at most this many content tags, highest scores first (default: config.yaml, else {hydrus_safety.DEFAULT_MAX_TAGS})")
def evaluate(filename, cpu, model, threshold, max_tags):
    config = load_config()
    settings = common_settings(config, cpu, model, threshold, max_tags)
    modelinfo, integerator = load_model(settings["model"], settings["cpu"])
    image = Image.open(filename)
    ratings, tag_scores = integerator.interrogate(image)
    rating, content, _ = build_tags(modelinfo, ratings, tag_scores, settings["threshold"], settings["max_tags"], False)
    click.echo("rating: " + (rating or "none"))
    click.echo("tags: " + ", ".join(content))


@click.command()
@click.argument("hash")
@click.option("--token", help="The API token for your Hydrus server")
@click.option("--cpu/--gpu", default=None, help="Run the model on the CPU or the GPU (default: config.yaml, else GPU)")
@click.option("--model", default=None, help=f"The tagging model folder to use (default: config.yaml, else {DEFAULT_MODEL})")
@click.option("--threshold", type=float, default=None, help=f"The threshold to drop tags below (default: config.yaml, else {DEFAULT_THRESHOLD})")
@click.option("--max-tags", type=int, default=None, help=f"Keep at most this many content tags, highest scores first (default: config.yaml, else {hydrus_safety.DEFAULT_MAX_TAGS})")
@click.option("--host", default=None, help=f"The URL for your Hydrus server (default: config.yaml, else {DEFAULT_HOST})")
@click.option("--tag-service", default=None, help=f"The local Hydrus tag service to add tags to (default: config.yaml, else '{DEFAULT_TAG_SERVICE}')")
@click.option("--ratings-only", is_flag=True, default=None, help="Only add the content rating tag (no content tags). Existing tags are never removed")
@click.option("--privacy/--show-tags", default=None, help="Hide or show the tag output in the console (default: config.yaml, else hide)")
@click.option("--dry-run", is_flag=True, help="Do everything except write to Hydrus, and print what would be sent")
@click.option("--allow-my-tags", is_flag=True, help="Allow writing to the built-in 'my tags' service")
def evaluate_api(hash, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy, dry_run, allow_my_tags):
    file_hash = hydrus_safety.normalize_hash(hash)
    if file_hash is None:
        raise click.UsageError(f"'{hash}' is not a sha256 hash (64 hex characters)")
    config = load_config()
    settings = api_settings(config, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy)

    modelinfo, integerator = load_model(settings["model"], settings["cpu"])
    if settings["ratings_only"] and not modelinfo['ratingsflag']:
        raise click.UsageError("--ratings-only set, but model does not support ratings!")
    client, service = connect(settings, allow_my_tags)

    metadata = client.get_file_metadata(hashes=[file_hash], only_return_basic_information=True, include_services_object=False)
    row, reason = hydrus_safety.metadata_row_for_hash(metadata, file_hash)
    if row is None:
        raise click.ClickException(f"Not tagging {file_hash}: {reason}")

    image_bytes = BytesIO(client.get_file(file_hash).content)
    image = Image.open(image_bytes)
    ratings, tag_scores = integerator.interrogate(image)
    rating, content, final_tags = build_tags(modelinfo, ratings, tag_scores, settings["threshold"], settings["max_tags"], settings["ratings_only"])

    if not settings["privacy"]:
        click.echo("rating: " + (rating or "none"))
        click.echo("tags: " + ", ".join(content))

    if dry_run:
        click.echo("DRY RUN, nothing written: " + describe_send(file_hash, service, final_tags, not settings["privacy"]))
        return
    client.add_tags(
        hashes=[file_hash],
        service_keys_to_tags={service[0]: final_tags},
        override_previously_deleted_mappings=False,
    )
    click.echo(f"Added {len(final_tags)} tags to {file_hash}")


@click.command()
@click.argument("inputfile")
@click.option("--token", help="The API token for your Hydrus server")
@click.option("--cpu/--gpu", default=None, help="Run the model on the CPU or the GPU (default: config.yaml, else GPU)")
@click.option("--model", default=None, help=f"The tagging model folder to use (default: config.yaml, else {DEFAULT_MODEL})")
@click.option("--threshold", type=float, default=None, help=f"The threshold to drop tags below (default: config.yaml, else {DEFAULT_THRESHOLD})")
@click.option("--max-tags", type=int, default=None, help=f"Keep at most this many content tags per file, highest scores first (default: config.yaml, else {hydrus_safety.DEFAULT_MAX_TAGS})")
@click.option("--host", default=None, help=f"The URL for your Hydrus server (default: config.yaml, else {DEFAULT_HOST})")
@click.option("--tag-service", default=None, help=f"The local Hydrus tag service to add tags to (default: config.yaml, else '{DEFAULT_TAG_SERVICE}')")
@click.option("--ratings-only", is_flag=True, default=None, help="Only add the content rating tag (no content tags). Existing tags are never removed")
@click.option("--privacy/--show-tags", default=None, help="Hide or show the tag output in the console (default: config.yaml, else hide)")
@click.option("--tags", "tags_mode", is_flag=True, help="Treat inputfile as a list of Hydrus search tags instead of hashes")
@click.option("--skip-resume", is_flag=True, help="Ignore any unfinished checkpoint and start a fresh batch")
@click.option("--skip-tagged", is_flag=True, help="Skip files that already have an AI-generated tags marker")
@click.option("--limit", type=int, default=None, help="Stop after this many files have been successfully tagged (skips do not count)")
@click.option("--dry-run", is_flag=True, help="Do everything except write to Hydrus; log what would be sent to dryrun.log")
@click.option("--allow-my-tags", is_flag=True, help="Allow writing to the built-in 'my tags' service")
def evaluate_api_batch(inputfile, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy, tags_mode, skip_resume, skip_tagged, limit, dry_run, allow_my_tags):
    interrogate.session_debug.clear()
    config = load_config()
    settings = api_settings(config, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy)
    threshold = settings["threshold"]
    max_tags = settings["max_tags"]
    ratings_only = settings["ratings_only"]
    privacy = settings["privacy"]

    if not os.path.isfile(inputfile):
        raise click.UsageError("input file not found!")

    if dry_run:
        # A dry run never touches the real checkpoint, so it can't affect resume.
        checkpoint_file = "dryrun.log"
        complete_marker = None
        skip_resume = True
        debug("DEBUG: --dry-run set; nothing will be written to Hydrus. Results go to dryrun.log")
    else:
        checkpoint_file = "session.log"
        complete_marker = "session.complete"
    resuming = False
    already_done = set()
    hashes = []
    # What this run will do. Written as the first line of a fresh session.log
    # and compared on resume, so a resume can't mix models, inputs or targets.
    current_run = resume_check.run_settings(
        settings["model"], inputfile, tags_mode, ratings_only,
        settings["tag_service_key"] or settings["tag_service"],
        threshold, max_tags, skip_tagged,
    )

    # Resume check loads what was already processed so we can exclude it
    # after the search. The log only contains hashes we actually attempted,
    # so we always re-query and then skip anything already in the log.
    if skip_resume:
        if not dry_run:
            debug("DEBUG: --skip-resume set; starting fresh batch")
    elif os.path.isfile(complete_marker):
        debug("DEBUG: previous batch completed; starting fresh")
    elif os.path.isfile(checkpoint_file):
        with open(checkpoint_file) as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        stored_limit = None
        hash_lines = []
        header = None
        for line in lines:
            if line.startswith(resume_check.HEADER_PREFIX):
                header = line
                continue
            if line.startswith("#limit:"):
                try:
                    stored_limit = int(line.split(":", 1)[1])
                except ValueError:
                    pass
                continue
            # Only real hashes count. Debug lines and comments are ignored.
            candidate = line.split("\t", 1)[0]
            if hydrus_safety.normalize_hash(candidate) == candidate.lower():
                hash_lines.append(line)
        already_done = {line.split("\t", 1)[0].lower() for line in hash_lines}
        ok_count = sum(1 for line in hash_lines if line.endswith("\tOK"))
        done_count = len(already_done)
        if done_count:
            # Refuse to resume if the unfinished batch used other settings.
            # This happens before the model loads or Hydrus is contacted.
            if header is None:
                click.echo(f"Warning: {checkpoint_file} has no run-settings line (made by an older version), so the resume can't be checked. Resuming as before.", err=True)
            else:
                try:
                    hard, soft = resume_check.compare(resume_check.parse_header(header), current_run)
                except ValueError:
                    hard, soft = [("run-settings line", "unreadable", "")], []
                if hard:
                    click.echo(resume_check.mismatch_message(hard, checkpoint_file), err=True)
                    # Exit code 0 on purpose: repeat-run.bat retries on any
                    # error, which would repeat this refusal forever.
                    click.get_current_context().exit(0)
                if soft:
                    click.echo(resume_check.soft_message(soft), err=True)
            resuming = True
            if stored_limit is not None:
                remaining_limit = max(0, stored_limit - ok_count)
                limit = remaining_limit
                debug(f"DEBUG: unfinished batch found — resuming with {done_count} done ({ok_count} OK), {remaining_limit} left of original limit {stored_limit}")
                if remaining_limit == 0:
                    debug("DEBUG: original limit already reached; nothing to do")
                    hashes = None  # sentinel: skip search and processing
            else:
                debug(f"DEBUG: unfinished batch found — resuming with {done_count} done")
        else:
            debug("DEBUG: checkpoint empty; starting fresh")
    else:
        debug("DEBUG: no previous batch found; starting fresh")

    modelinfo, integerator = load_model(settings["model"], settings["cpu"])
    if ratings_only and not modelinfo['ratingsflag']:
        raise click.UsageError("--ratings-only set, but model does not support ratings!")

    client, service = connect(settings, allow_my_tags)
    service_key = service[0]

    if hashes is None:
        raw_hashes = []
    elif tags_mode:
        with open(inputfile) as f:
            search_tags = [line.strip() for line in f if line.strip()]
        # --limit stops the run after N successful tags. It does not override
        # a system:limit the user put in the search. If they didn't set one,
        # add a safety cap so we don't pull the entire database.
        if limit:
            has_system_limit = any(t.lower().startswith("system:limit") for t in search_tags)
            if not has_system_limit:
                safety = limit * 20
                search_tags.append(f"system:limit is {safety}")
                debug(f"DEBUG: --limit {limit} set with no system:limit; adding safety cap of {safety}")
            else:
                debug(f"DEBUG: --limit {limit} set; keeping your system:limit")
        # Flash files (.swf/.fla/.flash) cannot be processed. Always exclude them
        # from tag searches so Hydrus never returns them.
        flash_filter = "system:filetype is not flash"
        if flash_filter not in search_tags:
            search_tags.append(flash_filter)
            debug("DEBUG: excluding flash files from search")
        # --skip-tagged: let Hydrus exclude already-tagged files instead of
        # fetching them and skipping after the fact. Uses the current model's
        # own marker so it works for any model the user has installed.
        if skip_tagged:
            marker = modelinfo['modelname'] + " ai generated tags"
            negative = "-" + marker
            if negative not in search_tags and marker not in search_tags:
                search_tags.append(negative)
                debug(f"DEBUG: --skip-tagged set; adding negative search term: {negative}")
        debug(f"DEBUG: Searching Hydrus for tags: {search_tags}")
        result = client.search_files(tags=search_tags, return_hashes=True, return_file_ids=False)
        if isinstance(result, dict):
            raw_hashes = result.get("hashes", [])
        else:
            raw_hashes = result
        debug(f"DEBUG: Found {len(raw_hashes)} files")
    else:
        with open(inputfile) as hashfile_f:
            raw_hashes = [line.strip() for line in hashfile_f if line.strip()]

    # Only lowercase 64-hex sha256 hashes go any further. "sha256:" prefixes
    # are removed, duplicates are dropped, and anything else is reported.
    hashes = []
    invalid = []
    seen = set()
    for raw in raw_hashes:
        file_hash = hydrus_safety.normalize_hash(raw)
        if file_hash is None:
            invalid.append(raw)
            continue
        if file_hash in seen:
            continue
        seen.add(file_hash)
        hashes.append(file_hash)
    if invalid:
        debug(f"DEBUG: ignored {len(invalid)} lines that are not sha256 hashes, e.g. {invalid[:3]}")

    # Exclude anything already in the log so resume continues cleanly.
    if already_done:
        before = len(hashes)
        hashes = [h for h in hashes if h not in already_done]
        debug(f"DEBUG: excluded {before - len(hashes)} already-logged hashes; {len(hashes)} candidates remaining")

    # Fresh batch: write the debug block, then the limit header. Hashes are
    # appended as they are actually processed, so the log never holds
    # unprocessed ones. On resume, append a separator and the new debug.
    if not resuming:
        if complete_marker and os.path.isfile(complete_marker):
            os.remove(complete_marker)
        debug(f"DEBUG: initialized {checkpoint_file}")
        with open(checkpoint_file, "w") as f:
            if not dry_run:
                f.write(resume_check.header_line(current_run) + "\n")
            for line in interrogate.session_debug:
                f.write(line + "\n")
            if limit:
                f.write(f"#limit:{limit}\n")
    else:
        with open(checkpoint_file, "a") as f:
            f.write("\n# resumed\n")
            for line in interrogate.session_debug:
                f.write(line + "\n")

    # Append a processed hash to the log. status is "OK" or a failure reason.
    def log_result(hash_value, status):
        try:
            with open(checkpoint_file, "a") as f:
                f.write(f"{hash_value}\t{status}\n")
        except Exception as e:
            debug(f"DEBUG: failed to update checkpoint for {hash_value}: {e}")

    tagged_count = 0
    skipped_count = 0

    # Collect every tag string out of a Hydrus metadata dict, whatever nesting it uses
    def collect_tags(obj):
        found = []
        if isinstance(obj, dict):
            for v in obj.values():
                found.extend(collect_tags(v))
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    found.append(item)
                else:
                    found.extend(collect_tags(item))
        return found

    # True if the file already carries an AI-generated marker for this model
    # (handles both spaced and underscored forms, and the "ratings only" variant)
    def already_ai_tagged(file_meta):
        if not file_meta:
            return False
        existing = collect_tags(file_meta.get("tags", {}))
        model_name = modelinfo['modelname'].lower().replace("_", " ")
        for tag in existing:
            normalized = tag.lower().replace("_", " ")
            if "ai generated tags" in normalized and model_name in normalized:
                return True
        return False

    # When --limit is set, the bar tracks successful tags against that limit.
    # Otherwise it tracks the full hash list and moves on every file.
    if limit:
        bar = click.progressbar(length=limit)
    else:
        bar = click.progressbar(length=len(hashes))
    with bar:
        for hash in hashes:
            if not limit:
                bar.update(1)
            click.echo(" processing: " + hash)

            # Metadata first: it confirms Hydrus really has this exact file
            # (non-null file_id) before anything is downloaded or written.
            # Full metadata (with tags) is only needed for --skip-tagged.
            try:
                metadata = client.get_file_metadata(
                    hashes=[hash],
                    only_return_basic_information=not skip_tagged,
                    include_services_object=False,
                )
            except Exception as e:
                log_result(hash, f"failed to fetch metadata from Hydrus: {e}")
                skipped_count += 1
                continue
            file_meta, reason = hydrus_safety.metadata_row_for_hash(metadata, hash)
            if file_meta is None:
                log_result(hash, f"skipped: {reason}")
                skipped_count += 1
                continue
            num_frames = file_meta.get("num_frames")
            ext = (file_meta.get("ext") or "").lower()

            # Skip files that already have this model's AI-generated marker
            if skip_tagged and already_ai_tagged(file_meta):
                log_result(hash, "already ai-tagged")
                skipped_count += 1
                continue

            # Skip formats that hang or aren't useful to tag
            if ext in (".swf", ".fla", ".flash"):
                log_result(hash, f"unsupported format {ext}")
                skipped_count += 1
                continue

            try:
                file = client.get_file(hash).content
            except Exception as e:
                log_result(hash, f"failed to fetch from Hydrus: {e}")
                skipped_count += 1
                continue

            if num_frames is None:
                try:
                    image = Image.open(BytesIO(file))
                    ratings, tag_scores = integerator.interrogate(image)
                    threshold_local = threshold
                except Exception as e:
                    log_result(hash, f"unsupported file: {e}")
                    skipped_count += 1
                    continue
            else:
                # Multi-frame (video / animated)
                if num_frames >= 7:
                    frames_indexes = np.round(np.linspace(0, num_frames, 7)[1:-1])
                else:
                    frames_indexes = [np.round(num_frames / 2)]
                ratings, tag_scores = {}, {}
                threshold_local = threshold * 2
                successful_frames = 0
                try:
                    for index in frames_indexes:
                        frame = None
                        last_error = None
                        # Prefer pyav when available; fall back to default backend
                        for plugin in ("pyav", None):
                            try:
                                kwargs = {"index": int(index)}
                                if plugin:
                                    kwargs["plugin"] = plugin
                                frame = iio.imread(BytesIO(file), extension=ext, **kwargs)
                                break
                            except Exception as e:  # noqa: BLE001 - try the next backend
                                last_error = e
                                continue
                        if frame is None:
                            click.echo(f"  could not read frame {int(index)} of {hash}: {last_error}")
                            continue
                        image = Image.fromarray(frame)
                        rate, tag = integerator.interrogate(image)
                        if rate:
                            for r in rate:
                                ratings[r] = ratings.get(r, 0) + rate[r]
                        for t in tag:
                            tag_scores[t] = tag_scores.get(t, 0) + tag[t]
                        successful_frames += 1
                    if successful_frames == 0:
                        log_result(hash, "failed to decode any frames")
                        skipped_count += 1
                        continue
                except Exception as e:
                    log_result(hash, f"failed to decode: {e}")
                    skipped_count += 1
                    continue

            rating, content, final_tags = build_tags(modelinfo, ratings, tag_scores, threshold_local, max_tags, ratings_only)

            if not privacy:
                click.echo("rating: " + (rating or "none"))
                click.echo("tags: " + ", ".join(content))
                click.echo()

            if dry_run:
                click.echo(" DRY RUN, nothing written: " + describe_send(hash, service, final_tags, not privacy))
                log_result(hash, "DRY RUN\t" + ", ".join(final_tags))
                tagged_count += 1
                if limit:
                    bar.update(1)
                if limit and tagged_count >= limit:
                    debug(f"DEBUG: reached --limit {limit}; stopping")
                    break
                continue

            # Write tags to Hydrus, then record the hash so we can resume later
            try:
                client.add_tags(
                    hashes=[hash],
                    service_keys_to_tags={service_key: final_tags},
                    override_previously_deleted_mappings=False,
                )
                # Only mark as done after a successful write
                log_result(hash, "OK")
                tagged_count += 1
                if limit:
                    bar.update(1)
                if limit and tagged_count >= limit:
                    debug(f"DEBUG: reached --limit {limit}; stopping")
                    break
            except Exception as e:
                log_result(hash, f"failed to add tags: {e}")
                skipped_count += 1
                continue

    if dry_run:
        debug(f"DEBUG: dry run complete — {tagged_count} files would be tagged, {skipped_count} skipped — see {checkpoint_file}")
        return

    # Batch finished cleanly. Strip the OK status from successful lines so the
    # log is a clean hash list for copy-paste. Failures keep their reason.
    try:
        with open(checkpoint_file) as f:
            lines = f.readlines()
        with open(checkpoint_file, "w") as f:
            for line in lines:
                stripped = line.rstrip("\n")
                if stripped.endswith("\tOK"):
                    f.write(stripped[:-3] + "\n")
                else:
                    f.write(line if line.endswith("\n") else line + "\n")
        debug("DEBUG: stripped OK status from successful hashes in session.log")
    except Exception as e:
        debug(f"DEBUG: failed to clean OK statuses: {e}")

    debug(f"DEBUG: batch complete — tagged {tagged_count} files, skipped {skipped_count} files — see session.log for details")
    # Mark the batch as finished so the next launch starts fresh by default.
    with open(complete_marker, "w") as f:
        f.write(f"tagged={tagged_count} skipped={skipped_count}\n")


if __name__ == '__main__':
    Image.init()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cli.add_command(evaluate)
    cli.add_command(evaluate_api)
    cli.add_command(evaluate_api_batch)
    cli(args=upgrade_legacy_bool_args(sys.argv[1:]))
