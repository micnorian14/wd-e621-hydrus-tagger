"""What the tagger shows in the console window.

By default: a few lines about the run, one progress bar that redraws in
place, a short line for each file that had a problem, and a summary.
With --verbose (or "verbose: true" in config.yaml) every DEBUG line and a
line per file are shown too. Nothing here changes what goes into
session.log or dryrun.log.
"""
import os
import re
import shutil
import sys
import time

import click

verbose = False

# Colours (display only), matching hydrus-video-deduplicator: Rich's blue,
# green, yellow and red, numbers in bold cyan like Rich's highlighter, and
# times in bright yellow. Plain ANSI codes, so no extra package is needed.
RESET = "\x1b[0m"
BLUE = "\x1b[94m"  # bright blue; plain blue (34) is too dark on a black cmd window
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
CYAN = "\x1b[1;36m"
BRIGHT_YELLOW = "\x1b[93m"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Numbers and quoted text, as Rich's ReprHighlighter colours them.
_HIGHLIGHT_RE = re.compile(r"(?P<str>'[^']*')|(?P<number>(?<![\w.])-?[0-9]+\.?[0-9]*\b)")

force_color = None  # tests set True/False; None = decide from the console
_vt_ok = None


def _enable_windows_vt():
    """Turn on ANSI colour support in the Windows console (Windows 10+)."""
    global _vt_ok
    if _vt_ok is None:
        _vt_ok = True
        if os.name == "nt":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                for handle_id in (-11, -12):  # stdout, stderr
                    handle = kernel32.GetStdHandle(handle_id)
                    mode = ctypes.c_uint32()
                    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                        continue
                    if not kernel32.SetConsoleMode(handle, mode.value | 0x0004):
                        _vt_ok = False
            except (AttributeError, OSError):
                _vt_ok = False
    return _vt_ok


def color_enabled(stream=None, err=False):
    """Colours only in a real console, and never when NO_COLOR is set."""
    if force_color is not None:
        return force_color
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    if stream is None:
        stream = sys.stderr if err else sys.stdout
    isatty = getattr(stream, "isatty", None)
    if not (isatty and isatty()):
        return False
    return _enable_windows_vt()


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def paint(text, color, err=False, stream=None):
    """Whole text in one colour (or plain, if colours are off)."""
    if not color_enabled(stream, err):
        return text
    return f"{color}{text}{RESET}"


def highlight(text, base, err=False, stream=None):
    """Text in a base colour, with numbers bold cyan and quoted text green."""
    if not color_enabled(stream, err):
        return text
    out, pos = [base], 0
    for m in _HIGHLIGHT_RE.finditer(text):
        out.append(text[pos:m.start()])
        out.append((GREEN if m.group("str") else CYAN) + m.group(0) + RESET + base)
        pos = m.end()
    out.append(text[pos:] + RESET)
    return "".join(out)


def info(label, value):
    """A startup line: blue label, highlighted value."""
    click.echo(highlight(f"{label:<8} {value}", BLUE))


def success(text):
    click.echo(highlight(text, GREEN))


def warn(text, err=False):
    click.echo(paint(text, YELLOW, err=err), err=err)


def install_quiet_interrupts():
    """Don't print "Exception ignored in ... __del__" noise after Ctrl+C.

    When Ctrl+C lands while a video is being read, imageio's reader is cleaned
    up later and Python reports the KeyboardInterrupt again as an "ignored"
    error. Only that case is silenced; any other ignored error still shows.
    """
    previous = sys.unraisablehook
    if getattr(previous, "_quiet_interrupts", False):
        return

    def hook(unraisable):
        if isinstance(unraisable.exc_value, KeyboardInterrupt):
            return
        previous(unraisable)

    hook._quiet_interrupts = True
    sys.unraisablehook = hook


def install_error_colors():
    """Show click's "Error: ..." messages in red. Called once at startup."""
    original = click.ClickException.format_message
    if getattr(original, "_red", False):
        return

    def red_message(self):
        return paint(original(self), RED, err=True)

    red_message._red = True
    click.ClickException.format_message = red_message


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

    def line(self, width, color=False):
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
        filled = int(bar_width * fraction) if bar_width >= 6 else 0
        if bar_width >= 6:
            text = f"{self.label} [{'#' * filled}{'-' * (bar_width - filled)}] {info}"
        else:
            text = f"{self.label} {info}"
        if len(text) > width or not color:
            return text[:width]
        # Same text with colours added; the visible width doesn't change.
        c = {"tagged": GREEN, "skipped": YELLOW, "failed": RED}
        counts = f"ok {c['tagged']}{self.counts['tagged']}{RESET}"
        if self.counts["skipped"]:
            counts += f" skip {c['skipped']}{self.counts['skipped']}{RESET}"
        if self.counts["failed"]:
            counts += f" fail {c['failed']}{self.counts['failed']}{RESET}"
        timing = re.sub(r"\d+:\d\d(:\d\d)?", lambda m: BRIGHT_YELLOW + m.group(0) + RESET, timing)
        info = (f"{CYAN}{pos}{RESET}/{CYAN}{self.total}{RESET} {CYAN}{int(fraction * 100)}%{RESET}"
                f" | {counts} | {timing}")
        if bar_width >= 6:
            bar = f"{GREEN}{'#' * filled}{RESET}{'-' * (bar_width - filled)}"
            return f"{self.label} [{bar}] {info}"
        return f"{self.label} {info}"

    def draw(self, force=False):
        if not self.interactive:
            return
        now = time.monotonic()
        if not force and self._last_draw is not None and now - self._last_draw < self.min_interval:
            return
        self._last_draw = now
        text = self.line(self.width or console_width(), color=color_enabled(self.stream))
        visible = len(strip_ansi(text))
        padding = " " * max(0, self._last_len - visible)
        self.stream.write("\r" + text + padding)
        self.stream.flush()
        self._last_len = visible
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
