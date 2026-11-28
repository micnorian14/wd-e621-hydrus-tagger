import json
import os.path
import sys
import time
from io import BytesIO

import click
import hydrus_api
import imageio.v3 as iio
import numpy as np
import yaml
from PIL import Image, ImageFile

from . import console, hydrus_safety, interrogate, resume_check
from .hydrus_safety import SafetyError

Image.MAX_IMAGE_PIXELS = None

DEFAULT_MODEL = "wd-v1-4-vit-tagger-v2"
DEFAULT_THRESHOLD = 0.35
DEFAULT_HOST = "http://127.0.0.1:45869"
DEFAULT_TAG_SERVICE = "A.I. Tags"
DEFAULT_INPUT = "input.txt"
MODES = ("auto", "tags", "hashes")

# Built-in retry: wait 3 s, then 6, 12, ... up to 60 s between tries. Give up
# after this many failures in a row with no file finished in between, so an
# error that happens every time (bad model, wrong token) can't loop forever.
RETRY_FIRST_WAIT = 3
RETRY_MAX_WAIT = 60
RETRY_MAX_IN_A_ROW = 8


def load_config(path="config.yaml"):
    """Load optional config.yaml from the current working directory."""
    if os.path.isfile(path):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            console.warn(f"Warning: could not load {path}: {e}", err=True)
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


def to_limit(value):
    """Read the limit setting. 0, blank or "none" means no limit."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none", "no", "off")):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise click.UsageError(f"limit must be a whole number, 0 for no limit (got {value!r})") from None
    if number < 0:
        raise click.UsageError(f"limit can't be negative (got {number})")
    return number


def read_input(path, mode="auto"):
    """Read the input file. Returns (mode, lines) with mode 'tags' or 'hashes'.

    Blank lines and lines starting with # are ignored. In auto mode the file is
    hashes if every line is a sha256 hash, tags if none are, and refused if mixed.
    """
    with open(path, encoding="utf-8-sig") as f:
        entries = [line.strip() for line in f]
    entries = [line for line in entries if line and not line.startswith("#")]
    name = os.path.basename(path)
    if not entries:
        raise click.UsageError(
            f"{name} has nothing to process. Add Hydrus search tags or file hashes, one per line "
            f"(see input.txt.example)."
        )
    if mode != "auto":
        return mode, entries
    is_hash = [hydrus_safety.normalize_hash(line) is not None for line in entries]
    if all(is_hash):
        return "hashes", entries
    if not any(is_hash):
        return "tags", entries
    first_hash = entries[is_hash.index(True)]
    first_tag = entries[is_hash.index(False)]
    raise click.UsageError(
        f"{name} mixes file hashes and search tags (for example '{first_hash[:20]}...' and '{first_tag}'). "
        f"Put only one kind in the file, or set 'mode: tags' or 'mode: hashes' in config.yaml (or use --mode)."
    )


class ResumeRefused(click.ClickException):
    """The unfinished batch was started with different settings."""


# Hydrus not answering, or busy: worth stopping the batch and retrying later.
CONNECTION_ERRORS = (hydrus_api.ConnectionError, hydrus_api.DatabaseLocked)
# Never retried: they would fail the same way every time, or the user asked to stop.
NO_RETRY = (
    click.exceptions.ClickException,
    click.exceptions.Exit,
    click.exceptions.Abort,
    SafetyError,
    hydrus_api.InsufficientAccess,
    hydrus_api.MissingParameter,
)


def describe_error(error):
    """Readable one-line description; hydrus_api hides the real cause behind __cause__."""
    cause = error.__cause__ if isinstance(error, hydrus_api.ConnectionError) and error.__cause__ else error
    return f"{type(cause).__name__}: {cause}"


# Older bat files and docs pass these as "--cpu 0", "--privacy true", etc.
# They are on/off flags now, so translate the old form before click sees it.
_LEGACY_BOOL_FLAGS = {
    "--cpu": ("--cpu", "--gpu"),
    "--privacy": ("--privacy", "--show-tags"),
    "--ratings-only": ("--ratings-only", "--no-ratings-only"),
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


def debug(msg):
    """Record a debug line for session.log; print it only with --verbose."""
    if console.verbose:
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


def apply_verbose(verbose, config):
    console.set_verbose(to_bool(resolve(verbose, "verbose", False, config)))


SERVICE_TYPES = {5: "local tag service", 0: "tag repository", 10: "all known tags"}


def describe_service(service):
    """'name' (local tag service). The key is only shown in full with --verbose."""
    kind = SERVICE_TYPES.get(service[2], f"type {service[2]}")
    if console.verbose:
        return f"'{service[1]}' ({kind}, key {service[0]})"
    return f"'{service[1]}' ({kind})"


def provider_name(integerator, cpu):
    """What the model really runs on, as onnxruntime reports it."""
    get_providers = getattr(getattr(integerator, "model", None), "get_providers", None)
    if get_providers is None:
        return "CPU" if cpu else "GPU"
    if "CUDAExecutionProvider" in get_providers():
        return "GPU (CUDA)"
    return "CPU" if cpu else "CPU (GPU not available)"


def common_settings(config, cpu, model, threshold, max_tags, keep_underscores=None):
    """Settings shared by every command: CLI flag, then config.yaml, then built-in default."""
    settings = {
        "cpu": to_bool(resolve(cpu, "cpu", False, config)),
        "model": resolve(model, "model", DEFAULT_MODEL, config),
        "threshold": resolve(threshold, "threshold", DEFAULT_THRESHOLD, config),
        "max_tags": resolve(max_tags, "max_tags", hydrus_safety.DEFAULT_MAX_TAGS, config),
        "keep_underscores": to_bool(resolve(keep_underscores, "keep_underscores", False, config)),
    }
    try:
        warning = hydrus_safety.check_threshold(settings["threshold"])
        settings["threshold"] = float(settings["threshold"])
        settings["max_tags"] = hydrus_safety.check_max_tags(settings["max_tags"])
    except SafetyError as e:
        raise click.UsageError(str(e)) from None
    if warning:
        console.warn(warning, err=True)
    return settings


def api_settings(config, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy, keep_underscores=None):
    """Settings for the Hydrus commands."""
    settings = common_settings(config, cpu, model, threshold, max_tags, keep_underscores=keep_underscores)
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

def build_model_marker(modelinfo, ratings_only=False):
    prefix = "ratings only " if ratings_only else ""
    marker = f"tagger:{prefix}{modelinfo['modelname']} (ai generated tags)"

    return hydrus_safety.sanitize_tags(
        [marker],
        keep_namespace=True,
        keep_underscores=False
    )[0]

def build_tags(modelinfo, ratings, tag_scores, threshold, max_tags, ratings_only, *, keep_underscores=False):
    """Return (rating, content_tags, final_tags). final_tags is exactly what gets sent."""
    rating = pick_rating(modelinfo, ratings)

    content = [] if ratings_only else hydrus_safety.sanitize_tags(
        hydrus_safety.select_tags(tag_scores, threshold, max_tags),
        keep_underscores=keep_underscores
    )

    own = []

    if rating:
        own.append("rating:" + rating)

    own.append(build_model_marker(modelinfo, ratings_only))

    final = list(dict.fromkeys(content + own))
    return rating, content, final


def describe_send(file_hash, service, final_tags, show_tags=True):
    text = f"{file_hash} -> {describe_service(service)}: {len(final_tags)} tags"
    if show_tags:
        text += ": " + ", ".join(final_tags)
    return text


class DefaultRunGroup(click.Group):
    """A command group that runs the 'run' command when no command is named.

    So "python -m wd-hydrus-tagger", "... --dry-run" and "... my-list.txt"
    (a file dropped on run.bat) all start a batch.
    """

    def parse_args(self, ctx, args):
        if not args or (args[0] not in self.commands and args[0] not in ctx.help_option_names):
            args = ["run", *args]
        return super().parse_args(ctx, args)


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}


@click.group(cls=DefaultRunGroup, context_settings=CONTEXT_SETTINGS)
def cli():
    """Tag Hydrus files with an AI tagging model.

    With no command, this runs a batch using the settings in config.yaml
    (the same as the "run" command). Any setting can be changed for one run
    with a flag, for example:

    \b
      python -m wd-hydrus-tagger --dry-run
      python -m wd-hydrus-tagger my-list.txt --limit 10

    Use "COMMAND --help" to see the options of a command.
    """


# Help text shared by several commands. "config: x" names the config.yaml
# setting that is used when the flag is not given.
HELP = {
    "token": "Hydrus Client API access key. [config: token]",
    "host": f"Address of the Hydrus Client API. [config: host, default {DEFAULT_HOST}]",
    "tag_service": f"Local Hydrus tag service to add tags to. [config: tag_service, default '{DEFAULT_TAG_SERVICE}']",
    "model": f"Model folder inside model/. [config: model, default {DEFAULT_MODEL}]",
    "threshold": f"Lowest score (0-1) a tag needs to be kept. [config: threshold, default {DEFAULT_THRESHOLD}]",
    "max_tags": f"Most content tags per file, best scores first. [config: max_tags, default {hydrus_safety.DEFAULT_MAX_TAGS}]",
    "cpu": "Run the model on the processor or the graphics card (GPU falls back to CPU if unavailable). [config: cpu, default --gpu]",
    "ratings_only": "Only add the rating tag, no content tags. Existing tags are never removed. [config: ratings_only, default off]",
    "privacy": "Hide or show each file's tags in the console. [config: privacy, default --privacy]",
    "allow_my_tags": "Allow writing to Hydrus's built-in 'my tags' service (blocked by default).",
    "keep_underscores": (
    "Preserve underscores in model-generated tags. "
    "Kaomojis are always preserved. "
    "[config: keep_underscores, default off]"),
    "verbose": "Show every detail: DEBUG lines, one line per file, full service key. [config: verbose, default off]",
}


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("filename")
@click.option("--model", default=None, help=HELP["model"])
@click.option("--threshold", type=float, default=None, help=HELP["threshold"])
@click.option("--max-tags", type=int, default=None, help=HELP["max_tags"])
@click.option("--cpu/--gpu", default=None, help=HELP["cpu"])
@click.option("--verbose/--no-verbose", default=None, help=HELP["verbose"])
@click.option(
    "--keep-underscores/--no-keep-underscores",
    default=None,
    help=HELP["keep_underscores"]
)
def evaluate(filename, cpu, model, threshold, max_tags, verbose, keep_underscores):
    """Tag one image file on disk and print the tags. Nothing is sent to Hydrus."""
    config = load_config()
    apply_verbose(verbose, config)
    settings = common_settings(config, cpu, model, threshold, max_tags, keep_underscores=keep_underscores)
    modelinfo, integerator = load_model(settings["model"], settings["cpu"])
    image = Image.open(filename)
    ratings, tag_scores = integerator.interrogate(image)
    rating, content, _ = build_tags(modelinfo, ratings, tag_scores, settings["threshold"], settings["max_tags"], False, keep_underscores=settings["keep_underscores"])
    click.echo("rating: " + (rating or "none"))
    click.echo("tags: " + ", ".join(content))


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("hash")
@click.option("--token", help=HELP["token"])
@click.option("--host", default=None, help=HELP["host"])
@click.option("--tag-service", default=None, help=HELP["tag_service"])
@click.option("--model", default=None, help=HELP["model"])
@click.option("--threshold", type=float, default=None, help=HELP["threshold"])
@click.option("--max-tags", type=int, default=None, help=HELP["max_tags"])
@click.option("--cpu/--gpu", default=None, help=HELP["cpu"])
@click.option("--ratings-only/--no-ratings-only", default=None, help=HELP["ratings_only"])
@click.option("--privacy/--show-tags", default=None, help=HELP["privacy"])
@click.option("--dry-run", is_flag=True, help="Do everything except write to Hydrus, and print what would be sent.")
@click.option("--allow-my-tags", is_flag=True, help=HELP["allow_my_tags"])
@click.option("--verbose/--no-verbose", default=None, help=HELP["verbose"])
@click.option(
    "--keep-underscores/--no-keep-underscores",
    default=None,
    help=HELP["keep_underscores"]
)
def evaluate_api(hash, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy, dry_run, allow_my_tags, verbose, keep_underscores):
    """Tag one file in Hydrus, given its sha256 hash."""
    file_hash = hydrus_safety.normalize_hash(hash)
    if file_hash is None:
        raise click.UsageError(f"'{hash}' is not a sha256 hash (64 hex characters)")
    config = load_config()
    apply_verbose(verbose, config)
    settings = api_settings(config, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy,keep_underscores=keep_underscores)

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
    rating, content, final_tags = build_tags(modelinfo, ratings, tag_scores, settings["threshold"], settings["max_tags"], settings["ratings_only"], keep_underscores=settings["keep_underscores"])

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


BATCH_PARAMS = [
    click.argument("input_path", metavar="[INPUT]", required=False),
    click.option("--mode", type=click.Choice(MODES, case_sensitive=False), default=None,
                 help="How to read INPUT: auto (works it out), tags (Hydrus search tags) or hashes (sha256 file hashes). [config: mode, default auto]"),
    click.option("--tags", "tags_flag", is_flag=True, help="Same as --mode tags (kept for older bat files)."),
    click.option("--model", default=None, help=HELP["model"]),
    click.option("--threshold", type=float, default=None, help=HELP["threshold"]),
    click.option("--max-tags", type=int, default=None, help=HELP["max_tags"]),
    click.option("--limit", type=int, default=None, help="Stop after this many files are tagged; skipped files don't count. 0 = no limit. [config: limit, default 0]"),
    click.option("--skip-tagged/--no-skip-tagged", default=None, help="Skip files this model has already tagged. [config: skip_tagged, default off]"),
    click.option("--ratings-only/--no-ratings-only", default=None, help=HELP["ratings_only"]),
    click.option("--dry-run/--no-dry-run", default=None, help="Do everything except write to Hydrus; what would be sent goes to dryrun.log. [config: dry_run, default off]"),
    click.option("--retry/--no-retry", default=None, help="If the batch crashes (e.g. Hydrus stops answering), wait and carry on where it left off. [config: retry, default off]"),
    click.option("--skip-resume", is_flag=True, help="Start a new batch even if the last one didn't finish (replaces session.log)."),
    click.option("--token", default=None, help=HELP["token"]),
    click.option("--host", default=None, help=HELP["host"]),
    click.option("--tag-service", default=None, help=HELP["tag_service"]),
    click.option("--cpu/--gpu", default=None, help=HELP["cpu"]),
    click.option("--privacy/--show-tags", default=None, help=HELP["privacy"]),
    click.option("--allow-my-tags", is_flag=True, help=HELP["allow_my_tags"]),
    click.option("--verbose/--no-verbose", default=None, help=HELP["verbose"]),
    click.option(
    "--keep-underscores/--no-keep-underscores", default=None, help=HELP["keep_underscores"]
),
]


def batch_params(f):
    for param in reversed(BATCH_PARAMS):
        f = param(f)
    return f


BATCH_HELP = """Tag a batch of Hydrus files listed in a text file.

INPUT is the text file to read (or drop one on run.bat). Without it the
"input" setting in config.yaml is used, else input.txt. The file holds
either Hydrus search tags or sha256 file hashes, one per line. Blank lines
and lines starting with # are ignored.

If a batch stops part way, the next run carries on from session.log.
"""


@click.command("run", context_settings=CONTEXT_SETTINGS, help=BATCH_HELP + "\nThis is what runs when no command is given.")
@batch_params
def run_cmd(**kwargs):
    run_batch_command(**kwargs)


@click.command("evaluate-api-batch", context_settings=CONTEXT_SETTINGS, short_help="Same as run (older name, kept for old bat files).", help=BATCH_HELP + "\nSame as \"run\"; kept so older bat files keep working.")
@batch_params
def evaluate_api_batch(**kwargs):
    run_batch_command(**kwargs)


def batch_settings(config, input_path, mode, tags_flag, model, threshold, max_tags, limit, skip_tagged,
                   ratings_only, dry_run, retry, skip_resume, token, host, tag_service, cpu, privacy, allow_my_tags, verbose, keep_underscores=None):
    """Every batch setting, from flags first, then config.yaml, then defaults.
    Also reads the input file, so mistakes stop the run before Hydrus is contacted."""
    apply_verbose(verbose, config)
    settings = api_settings(config, token, cpu, model, threshold, max_tags, host, tag_service, ratings_only, privacy, keep_underscores=keep_underscores)

    if tags_flag:
        if mode is not None and mode.lower() != "tags":
            raise click.UsageError(f"--tags and --mode {mode} disagree; use one of them")
        mode = "tags"
    mode = str(resolve(mode, "mode", "auto", config)).strip().lower()
    if mode not in MODES:
        raise click.UsageError(f"mode must be one of {', '.join(MODES)} (got '{mode}')")

    settings["input"] = str(input_path or resolve(None, "input", DEFAULT_INPUT, config))
    if not os.path.isfile(settings["input"]):
        hint = ""
        if input_path and not os.path.splitext(input_path)[1] and os.sep not in input_path and "/" not in input_path:
            hint = " If you meant a command, see --help."
        raise click.UsageError(
            f"Input file '{settings['input']}' not found. Create it (see input.txt.example), "
            f"set 'input' in config.yaml, or give the file path.{hint}"
        )
    settings["mode"], settings["entries"] = read_input(settings["input"], mode)

    settings["limit"] = to_limit(resolve(limit, "limit", 0, config))
    settings["skip_tagged"] = to_bool(resolve(skip_tagged, "skip_tagged", False, config))
    settings["dry_run"] = to_bool(resolve(dry_run, "dry_run", False, config))
    settings["retry"] = to_bool(resolve(retry, "retry", False, config))
    settings["skip_resume"] = bool(skip_resume)
    settings["allow_my_tags"] = bool(allow_my_tags)
    return settings


def run_batch_command(**kwargs):
    interrogate.session_debug.clear()
    config = load_config()
    settings = batch_settings(config, **kwargs)
    # Shared between attempts: how many files have been finished, whether this
    # run already started its own session.log, and the loaded model.
    state = {"files_done": 0, "log_started": False, "model": None, "announced": False}
    failures_in_a_row = 0
    while True:
        files_before = state["files_done"]
        try:
            run_batch(settings, settings["skip_resume"] and not state["log_started"], state)
            return
        except KeyboardInterrupt:
            if settings["dry_run"]:
                console.warn("Interrupted. Dry run results so far are in dryrun.log.", err=True)
            else:
                console.warn("Interrupted. Progress so far is saved in session.log; run again to carry on.", err=True)
            raise
        except NO_RETRY:
            raise
        except Exception as e:
            if not settings["retry"] or settings["dry_run"]:
                if isinstance(e, CONNECTION_ERRORS):
                    raise click.ClickException(
                        f"Lost contact with Hydrus ({describe_error(e)}). Run again to carry on where it stopped, or turn on retry."
                    ) from e
                raise
            if state["files_done"] > files_before:
                failures_in_a_row = 0
            failures_in_a_row += 1
            if failures_in_a_row >= RETRY_MAX_IN_A_ROW:
                raise click.ClickException(
                    f"Stopped retrying: the batch failed {failures_in_a_row} times in a row without finishing a file. "
                    f"Last error: {describe_error(e)}"
                ) from e
            wait = min(RETRY_FIRST_WAIT * 2 ** (failures_in_a_row - 1), RETRY_MAX_WAIT)
            console.warn(
                f"[{time.strftime('%H:%M:%S')}] The batch stopped with an error: {describe_error(e)}\n"
                f"Retrying in {wait} seconds ({failures_in_a_row} of {RETRY_MAX_IN_A_ROW - 1} retries in a row). Press Ctrl+C to stop.",
                err=True,
            )
            time.sleep(wait)
            interrogate.session_debug.clear()


def run_batch(settings, skip_resume, state):
    """One attempt at the batch. Raises on anything that should stop it."""
    interrogate.session_debug.clear()
    threshold = settings["threshold"]
    max_tags = settings["max_tags"]
    ratings_only = settings["ratings_only"]
    privacy = settings["privacy"]
    inputfile = settings["input"]
    tags_mode = settings["mode"] == "tags"
    skip_tagged = settings["skip_tagged"]
    limit = settings["limit"]
    dry_run = settings["dry_run"]
    allow_my_tags = settings["allow_my_tags"]
    debug(f"DEBUG: reading {inputfile} as {'search tags' if tags_mode else 'hashes'} ({len(settings['entries'])} lines)")
    started = time.monotonic()
    if not state["announced"]:
        state["announced"] = True
        lines_word = "line" if len(settings["entries"]) == 1 else "lines"
        console.info("Input:", f"{inputfile} ({'search tags' if tags_mode else 'file hashes'}, {len(settings['entries'])} {lines_word})")
        options = []
        if dry_run:
            options.append("DRY RUN (nothing is written to Hydrus)")
        if ratings_only:
            options.append("ratings only")
        if skip_tagged:
            options.append("skip tagged")
        if limit:
            options.append(f"limit {limit}")
        if settings["retry"] and not dry_run:
            options.append("retry on")
        if options:
            console.info("Options:", ", ".join(options))
    resume_note = None

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
    # and compared on resume, so a resume can't mix models, modes or targets.
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
                console.warn(f"Warning: {checkpoint_file} has no run-settings line (made by an older version), so the resume can't be checked. Resuming as before.", err=True)
            else:
                try:
                    hard, soft = resume_check.compare(resume_check.parse_header(header), current_run)
                except ValueError:
                    hard, soft = [("run-settings line", "unreadable", "")], []
                if hard:
                    raise ResumeRefused(resume_check.mismatch_message(hard, checkpoint_file))
                if soft:
                    console.warn(resume_check.soft_message(soft), err=True)
            resuming = True
            if stored_limit is not None:
                remaining_limit = max(0, stored_limit - ok_count)
                limit = remaining_limit
                debug(f"DEBUG: unfinished batch found — resuming with {done_count} done ({ok_count} OK), {remaining_limit} left of original limit {stored_limit}")
                resume_note = f"carrying on the unfinished batch in {checkpoint_file} ({done_count} done, {remaining_limit} left of the limit of {stored_limit})"
                if remaining_limit == 0:
                    debug("DEBUG: original limit already reached; nothing to do")
                    hashes = None  # sentinel: skip search and processing
            else:
                debug(f"DEBUG: unfinished batch found — resuming with {done_count} done")
                resume_note = f"carrying on the unfinished batch in {checkpoint_file} ({done_count} done)"
        else:
            debug("DEBUG: checkpoint empty; starting fresh")
    else:
        debug("DEBUG: no previous batch found; starting fresh")

    # The model is loaded once per run, not again on every retry.
    model_key = (settings["model"], settings["cpu"])
    if state.get("model") and state["model"][0] == model_key:
        modelinfo, integerator = state["model"][1]
    else:
        if resume_note:
            console.info("Resume:", resume_note)
            resume_note = None
        click.echo(console.paint(f"Loading model {settings['model']} ...", console.BLUE))
        modelinfo, integerator = load_model(settings["model"], settings["cpu"])
        state["model"] = (model_key, (modelinfo, integerator))
        console.info("Model:", f"{settings['model']} on {provider_name(integerator, settings['cpu'])}")
    if resume_note:
        console.info("Resume:", resume_note)
    if ratings_only and not modelinfo['ratingsflag']:
        raise click.UsageError("--ratings-only set, but model does not support ratings!")

    client, service = connect(settings, allow_my_tags)
    service_key = service[0]
    console.info("Target:", describe_service(service))

    expected_markers = frozenset({
        build_model_marker(modelinfo, ratings_only=False),
        build_model_marker(modelinfo, ratings_only=True)
    })

    if hashes is None:
        raw_hashes = []
    elif tags_mode:
        search_tags = list(settings["entries"])
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
        # Note: Files excluded in the tag search here won't be counted in the skip count.
        if skip_tagged:
            for marker in expected_markers:
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
        raw_hashes = list(settings["entries"])
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
    found_count = len(hashes)
    if already_done:
        before = len(hashes)
        hashes = [h for h in hashes if h not in already_done]
        debug(f"DEBUG: excluded {before - len(hashes)} already-logged hashes; {len(hashes)} candidates remaining")
    files_word = "file" if found_count == 1 else "files"
    found_text = f"{found_count} {files_word}"
    if found_count != len(hashes):
        found_text += f", {found_count - len(hashes)} already done, {len(hashes)} to go"
    if invalid:
        found_text += f" ({len(invalid)} lines that aren't hashes ignored)"
    console.info("Found:", found_text)

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
                # A retry after this point resumes this log instead of
                # starting over, even if --skip-resume was given.
                state["log_started"] = True
            for line in interrogate.session_debug:
                f.write(line + "\n")
            if limit:
                f.write(f"#limit:{limit}\n")
    else:
        with open(checkpoint_file, "a") as f:
            f.write("\n# resumed\n")
            for line in interrogate.session_debug:
                f.write(line + "\n")

    # One progress bar for the whole batch. With --limit it counts tagged
    # files toward the limit, otherwise every file.
    progress = console.Progress(
        limit if limit else len(hashes),
        label="Dry run" if dry_run else "Tagging",
        count_tagged_only=bool(limit),
    )
    frames_unread = 0
    hit_limit = False

    def detail(msg):
        """A per-file line, only shown with --verbose."""
        if console.verbose:
            progress.message(msg)

    # Append a processed hash to the log. status is "OK" or a failure reason.
    def log_result(hash_value, status):
        try:
            with open(checkpoint_file, "a") as f:
                f.write(f"{hash_value}\t{status}\n")
            state["files_done"] += 1
        except Exception as e:
            debug(f"DEBUG: failed to update checkpoint for {hash_value}: {e}")
        # Console only: count it, and show a short line for real problems.
        if status == "OK" or status.startswith("DRY RUN"):
            progress.update("tagged")
            return
        if status == "already ai-tagged" or status.startswith("unsupported format"):
            progress.update("skipped")
            detail(f"  {hash_value}: {status}")
            return
        progress.update("skipped" if status.startswith("skipped:") else "failed")
        if console.verbose:
            progress.message(f"  {hash_value}: {status}")
        else:
            short = " ".join(status.split())
            width = console.console_width() - 12
            if len(short) > width:
                short = short[:max(0, width - 3)] + "..."
            progress.message(console.paint(f"  {console.short_hash(hash_value)}  {short}", console.YELLOW, stream=progress.stream))

    def summary():
        took = console.duration(time.monotonic() - started)
        counts = progress.counts
        done_word = "would be tagged" if dry_run else "tagged"
        console.success(
            f"{'Dry run done' if dry_run else 'Done'} in {took}: {counts['tagged']} {done_word}, "
            f"{counts['skipped']} skipped, {counts['failed']} failed."
        )
        if hit_limit:
            console.success(f"Stopped at the limit of {limit} tagged files.")
        if frames_unread:
            console.warn(f"{frames_unread} video frames could not be read (use --verbose to see them).")
        if dry_run:
            console.success(f"Nothing was written to Hydrus. Details: {checkpoint_file}")
        else:
            console.success(f"Details: {checkpoint_file}")

    tagged_count = 0
    skipped_count = 0

    # True if the file already carries an AI-generated marker for this model
    def already_ai_tagged(file_meta):
        if not file_meta:
            return False

        current_tags = (
            file_meta
            .get("tags", {})
            .get(service_key, {})
            .get("storage_tags", {})
            .get("0", [])
        )

        # Return True if either expected marker is currently present.
        return not expected_markers.isdisjoint(current_tags)

    with progress:
        for hash in hashes:
            detail(" processing: " + hash)

            # Metadata first: it confirms Hydrus really has this exact file
            # (non-null file_id) before anything is downloaded or written.
            # Full metadata (with tags) is only needed for --skip-tagged.
            try:
                metadata = client.get_file_metadata(
                    hashes=[hash],
                    only_return_basic_information=not skip_tagged,
                    include_services_object=False,
                )
            except CONNECTION_ERRORS:
                # Hydrus is gone or busy: stop the batch (and retry, if on)
                # rather than marking every remaining file as failed.
                raise
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
            except CONNECTION_ERRORS:
                # Hydrus is gone or busy: stop the batch (and retry, if on)
                # rather than marking every remaining file as failed.
                raise
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
                            frames_unread += 1
                            detail(f"  could not read frame {int(index)} of {hash}: {last_error}")
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

            rating, content, final_tags = build_tags(modelinfo, ratings, tag_scores, threshold_local, max_tags, ratings_only, keep_underscores=settings["keep_underscores"])

            if not privacy:
                if console.verbose:
                    progress.message("rating: " + (rating or "none"))
                    progress.message("tags: " + ", ".join(content))
                    progress.message("")
                else:
                    progress.message(f"  {console.short_hash(hash)}  rating: {rating or 'none'} | tags: {', '.join(content)}")

            if dry_run:
                detail(" DRY RUN, nothing written: " + describe_send(hash, service, final_tags, not privacy))
                log_result(hash, "DRY RUN\t" + ", ".join(final_tags))
                tagged_count += 1
                if limit and tagged_count >= limit:
                    debug(f"DEBUG: reached --limit {limit}; stopping")
                    hit_limit = True
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
                if limit and tagged_count >= limit:
                    debug(f"DEBUG: reached --limit {limit}; stopping")
                    hit_limit = True
                    break
            except CONNECTION_ERRORS:
                # Hydrus is gone or busy: stop the batch (and retry, if on)
                # rather than marking every remaining file as failed.
                raise
            except Exception as e:
                log_result(hash, f"failed to add tags: {e}")
                skipped_count += 1
                continue

    if dry_run:
        debug(f"DEBUG: dry run complete — {tagged_count} files would be tagged, {skipped_count} skipped — see {checkpoint_file}")
        summary()
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
    summary()


cli.add_command(run_cmd)
cli.add_command(evaluate)
cli.add_command(evaluate_api)
cli.add_command(evaluate_api_batch)


if __name__ == '__main__':
    console.install_error_colors()
    console.install_quiet_interrupts()
    Image.init()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cli(args=upgrade_legacy_bool_args(sys.argv[1:]), prog_name="python -m wd-hydrus-tagger")
