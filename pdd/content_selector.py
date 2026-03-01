"""
Content selector module for precise extraction of file content.

Provides extraction based on line ranges, AST structures (Python),
Markdown sections, and regex patterns. Used by the PDD preprocessor
for selective includes.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.theme import Theme

# Rich console with custom theme for error reporting
_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "green",
        "path": "dim blue",
        "selector": "bold magenta",
    }
)
console = Console(theme=_theme)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SelectorError(Exception):
    """Raised when a selector is malformed or cannot be resolved."""


# ---------------------------------------------------------------------------
# Internal data helpers
# ---------------------------------------------------------------------------

@dataclass
class _Span:
    """A half-open range of 0-based line indices [start, end)."""
    start: int
    end: int


def _lines_of(content: str) -> list[str]:
    return content.splitlines(keepends=False) if hasattr(str, "splitlines") else content.split("\n")


def _splitlines(content: str) -> list[str]:
    """Split content into lines *without* trailing newline artifacts."""
    return content.splitlines()


def _extract_spans(lines: list[str], spans: list[_Span]) -> str:
    """Merge possibly-overlapping spans and return the selected text."""
    if not spans:
        return ""
    # Sort and merge
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[_Span] = [spans[0]]
    for sp in spans[1:]:
        prev = merged[-1]
        if sp.start <= prev.end:
            prev.end = max(prev.end, sp.end)
        else:
            merged.append(sp)
    selected: list[str] = []
    for sp in merged:
        selected.extend(lines[sp.start : sp.end])
    return "\n".join(selected)


# ---------------------------------------------------------------------------
# Selector parsers
# ---------------------------------------------------------------------------

_SELECTOR_RE = re.compile(
    r"^(?P<kind>lines|def|class|section|pattern):(?P<value>.+)$"
)


@dataclass
class _ParsedSelector:
    kind: str
    value: str


def _parse_selectors(selectors: list[str]) -> list[_ParsedSelector]:
    """Parse a list of selector strings into structured objects."""
    parsed: list[_ParsedSelector] = []
    for raw in selectors:
        raw = raw.strip()
        if not raw:
            continue
        m = _SELECTOR_RE.match(raw)
        if not m:
            raise SelectorError(
                f"Malformed selector: '{raw}'. "
                "Expected format: lines:N-M | def:name | class:Name[.method] | section:Heading | pattern:/regex/"
            )
        parsed.append(_ParsedSelector(kind=m.group("kind"), value=m.group("value")))
    return parsed


# ---------------------------------------------------------------------------
# Line selector
# ---------------------------------------------------------------------------

def _resolve_lines(content_lines: list[str], value: str) -> list[_Span]:
    """Resolve a ``lines:`` selector value into spans.

    Supported forms (1-based):
      ``N``        – single line
      ``N-M``      – inclusive range
      ``N-``       – from N to end
      ``-M``       – from start to M
    """
    total = len(content_lines)
    spans: list[_Span] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            # Could be N-M, N-, -M
            idx = part.index("-")
            left = part[:idx].strip()
            right = part[idx + 1 :].strip()
            if left == "" and right == "":
                raise SelectorError(f"Invalid line range: '{part}'")
            start_0 = (int(left) - 1) if left else 0
            end_0 = int(right) if right else total  # end is exclusive
            if start_0 < 0:
                start_0 = 0
            if end_0 > total:
                end_0 = total
            if start_0 >= end_0:
                raise SelectorError(
                    f"Empty or inverted line range: '{part}' (resolved {start_0+1}-{end_0})"
                )
            spans.append(_Span(start_0, end_0))
        else:
            n = int(part)
            if n < 1 or n > total:
                raise SelectorError(
                    f"Line {n} out of range (file has {total} lines)"
                )
            spans.append(_Span(n - 1, n))
    return spans


# ---------------------------------------------------------------------------
# AST helpers (Python)
# ---------------------------------------------------------------------------

def _node_start_line(node: ast.AST) -> int:
    """Return the 0-based start line of a node, including decorators."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno - 1
    return node.lineno - 1


def _node_end_line(node: ast.AST) -> int:
    """Return the 0-based exclusive end line of a node."""
    return node.end_lineno  # end_lineno is 1-based inclusive → use as exclusive 0-based


def _find_ast_node(
    tree: ast.Module,
    kind: str,
    value: str,
) -> list[_Span]:
    """Find spans for ``def:name`` or ``class:Name[.method]`` selectors."""
    spans: list[_Span] = []

    if kind == "def":
        # Top-level or nested function
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == value:
                    spans.append(_Span(_node_start_line(node), _node_end_line(node)))
        if not spans:
            raise SelectorError(f"Function '{value}' not found in source")

    elif kind == "class":
        if "." in value:
            cls_name, method_name = value.split(".", 1)
        else:
            cls_name = value
            method_name = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                if method_name is None:
                    spans.append(_Span(_node_start_line(node), _node_end_line(node)))
                else:
                    found_method = False
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name == method_name:
                                spans.append(
                                    _Span(_node_start_line(child), _node_end_line(child))
                                )
                                found_method = True
                    if not found_method:
                        raise SelectorError(
                            f"Method '{method_name}' not found in class '{cls_name}'"
                        )
        if not spans:
            target = f"Class '{cls_name}'" if method_name is None else f"Class '{cls_name}' (for method '{method_name}')"
            raise SelectorError(f"{target} not found in source")

    return spans


# ---------------------------------------------------------------------------
# Interface mode (Python)
# ---------------------------------------------------------------------------

def _interface_for_node(node: ast.AST, source_lines: list[str]) -> list[str]:
    """Produce interface-mode output for a single class or function node."""
    lines: list[str] = []

    if isinstance(node, ast.ClassDef):
        # Decorators
        for dec in node.decorator_list:
            lines.extend(source_lines[dec.lineno - 1 : dec.end_lineno])
        # Class definition line(s)
        class_start = node.lineno - 1
        # Find the colon that ends the class header
        class_header_end = class_start
        for i in range(class_start, node.end_lineno):
            if ":" in source_lines[i]:
                class_header_end = i
                break
        lines.extend(source_lines[class_start : class_header_end + 1])

        # Docstring
        ds = _extract_docstring_lines(node, source_lines)
        if ds:
            lines.extend(ds)

        # Methods (excluding private except __init__)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith("_") and child.name != "__init__":
                    continue
                lines.extend(_interface_for_func(child, source_lines))
            elif isinstance(child, ast.AnnAssign):
                # Class-level annotated assignments (type hints)
                lines.extend(source_lines[child.lineno - 1 : child.end_lineno])

    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        lines.extend(_interface_for_func(node, source_lines))

    return lines


def _interface_for_func(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
) -> list[str]:
    """Return interface lines for a single function/method."""
    lines: list[str] = []
    # Decorators
    for dec in node.decorator_list:
        lines.extend(source_lines[dec.lineno - 1 : dec.end_lineno])

    # Signature – may span multiple lines
    sig_start = node.lineno - 1
    # Find the colon ending the signature
    sig_end = sig_start
    paren_depth = 0
    for i in range(sig_start, node.end_lineno):
        line = source_lines[i]
        for ch in line:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
        if paren_depth <= 0 and ":" in line:
            sig_end = i
            break
    lines.extend(source_lines[sig_start : sig_end + 1])

    # Docstring
    ds = _extract_docstring_lines(node, source_lines)
    if ds:
        lines.extend(ds)

    # Determine indentation for the ellipsis
    body_indent = _body_indent(node, source_lines)
    lines.append(f"{body_indent}...")

    return lines


def _extract_docstring_lines(
    node: ast.AST, source_lines: list[str]
) -> list[str] | None:
    """If the first statement is a string constant (docstring), return its source lines."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, (ast.Constant,)):
        if isinstance(first.value.value, str):
            return source_lines[first.value.lineno - 1 : first.value.end_lineno]
    return None


def _body_indent(node: ast.AST, source_lines: list[str]) -> str:
    """Determine the indentation of the body of a function/class."""
    body = getattr(node, "body", None)
    if body:
        first_body_line = source_lines[body[0].lineno - 1]
        return first_body_line[: len(first_body_line) - len(first_body_line.lstrip())]
    # Fallback: indent from the node line + 4 spaces
    node_line = source_lines[node.lineno - 1]
    base = node_line[: len(node_line) - len(node_line.lstrip())]
    return base + "    "


def _full_interface(content: str, source_lines: list[str]) -> str:
    """Produce interface-mode output for an entire Python file."""
    tree = ast.parse(content)
    result_lines: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            result_lines.extend(source_lines[node.lineno - 1 : node.end_lineno])
        elif isinstance(node, ast.Assign):
            # Module-level constants
            result_lines.extend(source_lines[node.lineno - 1 : node.end_lineno])
        elif isinstance(node, ast.AnnAssign):
            result_lines.extend(source_lines[node.lineno - 1 : node.end_lineno])
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and node.name != "__init__":
                continue
            iface = _interface_for_node(node, source_lines)
            if result_lines and result_lines[-1].strip() != "":
                result_lines.append("")
            result_lines.extend(iface)
        elif isinstance(node, ast.Expr) and isinstance(
            getattr(node, "value", None), ast.Constant
        ):
            # Module-level docstring
            if isinstance(node.value.value, str):
                result_lines.extend(source_lines[node.lineno - 1 : node.end_lineno])

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Markdown section selector
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _resolve_section(content_lines: list[str], heading_text: str) -> list[_Span]:
    """Find a Markdown section by heading text.

    Returns all content under the heading until the next heading of the
    same or higher (fewer ``#``) level.
    """
    spans: list[_Span] = []
    i = 0
    while i < len(content_lines):
        m = _MD_HEADING_RE.match(content_lines[i])
        if m and m.group(2).strip() == heading_text.strip():
            level = len(m.group(1))
            start = i
            j = i + 1
            while j < len(content_lines):
                m2 = _MD_HEADING_RE.match(content_lines[j])
                if m2 and len(m2.group(1)) <= level:
                    break
                j += 1
            spans.append(_Span(start, j))
            i = j
        else:
            i += 1
    if not spans:
        raise SelectorError(f"Markdown section '{heading_text}' not found")
    return spans


# ---------------------------------------------------------------------------
# Regex pattern selector
# ---------------------------------------------------------------------------

def _resolve_pattern(content_lines: list[str], value: str) -> list[_Span]:
    """Select lines matching ``pattern:/regex/``."""
    # Strip surrounding slashes if present
    pattern = value.strip()
    if pattern.startswith("/") and pattern.endswith("/") and len(pattern) >= 2:
        pattern = pattern[1:-1]
    if not pattern:
        raise SelectorError("Empty regex pattern")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise SelectorError(f"Invalid regex pattern '{pattern}': {exc}") from exc

    spans: list[_Span] = []
    for i, line in enumerate(content_lines):
        if compiled.search(line):
            spans.append(_Span(i, i + 1))
    if not spans:
        raise SelectorError(f"No lines matched pattern '{pattern}'")
    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ContentSelector:
    """Precise content extraction from file content."""

    @staticmethod
    def select(
        content: str,
        selectors: list[str] | str,
        file_path: str | None = None,
        mode: str = "full",
    ) -> str:
        """Select portions of *content* according to *selectors*.

        Parameters
        ----------
        content:
            The full text content to select from.
        selectors:
            A list of selector strings **or** a single comma-separated string.
            Each selector has the form ``kind:value`` where *kind* is one of
            ``lines``, ``def``, ``class``, ``section``, ``pattern``.
        file_path:
            Optional file path used to infer the file type (e.g. ``.py``,
            ``.md``).  When ``None``, AST-based selectors will attempt to
            parse as Python.
        mode:
            ``"full"`` (default) returns the selected content verbatim.
            ``"interface"`` (Python only) returns signatures, docstrings,
            and type hints with bodies replaced by ``...``.

        Returns
        -------
        str
            The selected (and possibly transformed) content.
        """
        # Normalise selectors to a list
        if isinstance(selectors, str):
            selectors = [s.strip() for s in selectors.split(",") if s.strip()]

        if not selectors and mode == "interface":
            # No selectors but interface mode → produce interface for whole file
            source_lines = _splitlines(content)
            try:
                return _full_interface(content, source_lines)
            except SyntaxError as exc:
                _report_error(f"Failed to parse Python source: {exc}", file_path)
                raise SelectorError(f"Python parse error: {exc}") from exc

        if not selectors:
            return content

        parsed = _parse_selectors(selectors)
        source_lines = _splitlines(content)

        # Determine file type
        is_python = _is_python(file_path)
        is_markdown = _is_markdown(file_path)

        # We may need the AST for Python selectors
        tree: ast.Module | None = None
        needs_ast = any(p.kind in ("def", "class") for p in parsed)
        if needs_ast:
            if not is_python and file_path is not None:
                _report_error(
                    f"AST selectors (def/class) require a Python file, got '{file_path}'",
                    file_path,
                )
                raise SelectorError(
                    f"AST selectors require a .py file, got '{file_path}'"
                )
            try:
                tree = ast.parse(content)
            except SyntaxError as exc:
                _report_error(f"Failed to parse Python source: {exc}", file_path)
                raise SelectorError(f"Python parse error: {exc}") from exc

        needs_md = any(p.kind == "section" for p in parsed)
        if needs_md and not is_markdown and file_path is not None:
            _report_error(
                f"Section selector requires a Markdown file, got '{file_path}'",
                file_path,
            )
            raise SelectorError(
                f"Section selector requires a .md file, got '{file_path}'"
            )

        # Collect spans
        all_spans: list[_Span] = []
        for sel in parsed:
            try:
                if sel.kind == "lines":
                    all_spans.extend(_resolve_lines(source_lines, sel.value))
                elif sel.kind in ("def", "class"):
                    assert tree is not None
                    all_spans.extend(_find_ast_node(tree, sel.kind, sel.value))
                elif sel.kind == "section":
                    all_spans.extend(_resolve_section(source_lines, sel.value))
                elif sel.kind == "pattern":
                    all_spans.extend(_resolve_pattern(source_lines, sel.value))
                else:
                    raise SelectorError(f"Unknown selector kind: '{sel.kind}'")
            except SelectorError:
                raise
            except Exception as exc:
                _report_error(
                    f"Error processing selector '{sel.kind}:{sel.value}': {exc}",
                    file_path,
                )
                raise SelectorError(
                    f"Error processing selector '{sel.kind}:{sel.value}': {exc}"
                ) from exc

        if not all_spans:
            return ""

        # Interface mode post-processing for AST selectors
        if mode == "interface" and is_python and tree is not None:
            return _interface_from_spans(content, source_lines, tree, all_spans)

        return _extract_spans(source_lines, all_spans)


# ---------------------------------------------------------------------------
# Interface mode with specific spans
# ---------------------------------------------------------------------------

def _interface_from_spans(
    content: str,
    source_lines: list[str],
    tree: ast.Module,
    spans: list[_Span],
) -> str:
    """Produce interface output only for AST nodes overlapping *spans*."""
    # Merge spans
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[_Span] = [_Span(spans[0].start, spans[0].end)]
    for sp in spans[1:]:
        prev = merged[-1]
        if sp.start <= prev.end:
            prev.end = max(prev.end, sp.end)
        else:
            merged.append(sp)

    def _overlaps(node_span: _Span) -> bool:
        for sp in merged:
            if node_span.start < sp.end and node_span.end > sp.start:
                return True
        return False

    result_lines: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            ns = _Span(_node_start_line(node), _node_end_line(node))
            if _overlaps(ns):
                iface = _interface_for_node(node, source_lines)
                if result_lines and result_lines[-1].strip() != "":
                    result_lines.append("")
                result_lines.extend(iface)

    return "\n".join(result_lines) if result_lines else _extract_spans(source_lines, merged)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_python(file_path: str | None) -> bool:
    if file_path is None:
        return True  # assume Python when unknown
    return file_path.rstrip().lower().endswith(".py")


def _is_markdown(file_path: str | None) -> bool:
    if file_path is None:
        return False
    lower = file_path.rstrip().lower()
    return lower.endswith(".md") or lower.endswith(".markdown")


def _report_error(message: str, file_path: str | None = None) -> None:
    """Print a formatted error to the rich console."""
    location = f" in [path]{file_path}[/path]" if file_path else ""
    console.print(f"[error]ContentSelector error{location}:[/error] {message}")