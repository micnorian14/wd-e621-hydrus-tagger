"""What the tagger shows in the console window.

By default: a few lines about the run, one progress bar that redraws in
place, a short line for each file that had a problem, and a summary.
With --verbose (or "verbose: true" in config.yaml) every DEBUG line and a
line per file are shown too. Nothing here changes what goes into
session.log or dryrun.log.
"""
import shutil
import time

import click

verbose = False


def set_verbose(value):
    global verbose
    verbose = bool(value)


def duration(seconds):
    """1:05 or 2:03:04."""
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def short_hash(value):
    """First 8 characters, enough to find a file (or service key) again."""
    return str(value)[:8]


def console_width():
    # One less than the window, so a full line never wraps in Windows cmd.
    return max(20, shutil.get_terminal_size((80, 20)).columns - 1)


class Progress:
    """A one-line progress bar that redraws in place.

    Lines printed with message() appear above the bar. The bar is only drawn
    when the output is a real console; when it is redirected to a file or a
    pipe, only the messages are written.
    """

    def __init__(self, total, label="Tagging", count_tagged_only=False, stream=None, interactive=None,
                 min_interval=0.25, width=None):
        self.total = max(0, int(total))
        self.label = label
        # With a limit, the bar counts tagged files toward the limit.
        self.count_tagged_only = count_tagged_only
        self.stream = stream if stream is not None else click.get_text_stream("stdout")
        if interactive is None:
            isatty = getattr(self.stream, "isatty", None)
            interactive = bool(isatty and isatty())
        self.interactive = interactive
        self.min_interval = min_interval
        self.width = width
        self.started = time.monotonic()
        self.done = 0
        self.counts = {"tagged": 0, "skipped": 0, "failed": 0}
        self._last_draw = None
        self._last_len = 0
        self._drawn = False

    def __enter__(self):
        self.draw(force=True)
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def update(self, outcome):
        """Count one finished file: outcome is 'tagged', 'skipped' or 'failed'."""
        self.done += 1
        self.counts[outcome] += 1
        self.draw()

    def position(self):
        return self.counts["tagged"] if self.count_tagged_only else self.done

    def line(self, width):
        pos = self.position()
        if self.total:
            pos = min(pos, self.total)
        fraction = pos / self.total if self.total else 1.0
        elapsed = time.monotonic() - self.started
        if pos and pos < self.total:
            timing = f"{duration(elapsed)} (~{duration(elapsed / pos * (self.total - pos))} left)"
        else:
            timing = duration(elapsed)
        counts = f"ok {self.counts['tagged']}"
        if self.counts["skipped"]:
            counts += f" skip {self.counts['skipped']}"
        if self.counts["failed"]:
            counts += f" fail {self.counts['failed']}"
        info = f"{pos}/{self.total} {int(fraction * 100)}% | {counts} | {timing}"
        bar_width = min(30, width - len(self.label) - len(info) - 4)
        if bar_width >= 6:
            filled = int(bar_width * fraction)
            text = f"{self.label} [{'#' * filled}{'-' * (bar_width - filled)}] {info}"
        else:
            text = f"{self.label} {info}"
        return text[:width]

    def draw(self, force=False):
        if not self.interactive:
            return
        now = time.monotonic()
        if not force and self._last_draw is not None and now - self._last_draw < self.min_interval:
            return
        self._last_draw = now
        text = self.line(self.width or console_width())
        padding = " " * max(0, self._last_len - len(text))
        self.stream.write("\r" + text + padding)
        self.stream.flush()
        self._last_len = len(text)
        self._drawn = True

    def _clear(self):
        if self._drawn:
            self.stream.write("\r" + " " * self._last_len + "\r")
            self.stream.flush()

    def message(self, text, err=False):
        """Print a line above the bar."""
        self._clear()
        if err:
            click.echo(text, err=True)
        else:
            click.echo(text, file=self.stream)
        if self._drawn:
            self.draw(force=True)

    def close(self):
        if self._drawn:
            self.draw(force=True)
            self.stream.write("\n")
            self.stream.flush()
            self._drawn = False
