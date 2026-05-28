"""Treemap layout and tree utilities."""

from __future__ import annotations

from typing import Any, Sequence

import distinctipy
import numpy as np
import squarify  # type: ignore[import-untyped]
from numpy.typing import ArrayLike

from textual_plot.color_utils import (
    parse_style_to_rgb,
    rgb_style,
    rgb_too_close,
    tint_with_hue,
)

# 1-character border around children so parent is visible and selectable
TREEMAP_PARENT_PAD = 1
# Top inset for parent labels (3 rows); sides/bottom stay 1
TREEMAP_PARENT_LABEL_TOP = 3

TreemapNode: type = dict[str, Any]


def treemap_max_depth(nodes: list[TreemapNode]) -> int:
    """Max nesting depth of tree (1 = no nesting, 2 = one level of children, etc)."""
    if not nodes:
        return 0
    return 1 + max(treemap_max_depth(n.get("children") or []) for n in nodes)


def normalize_treemap_tree(
    values: ArrayLike | list[Any] | list[list[float]],
    labels: Sequence[str] | Sequence[Sequence[str]] | None = None,
) -> tuple[list[TreemapNode], bool]:
    """Normalize treemap input to a tree of nodes. Returns (nodes, is_nested)."""
    if not values:
        return ([], False)

    try:
        vals_list = list(values)
    except TypeError:
        vals_list = np.array(values).tolist()
    first = vals_list[0] if len(vals_list) > 0 else None

    # Flat: list of numbers
    if isinstance(first, (int, float, np.floating, np.integer)):
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
        n = len(arr)
        labels_list = list(labels)[:n] if labels else None
        nodes = [
            {
                "value": float(arr[i]),
                "label": (
                    labels_list[i]
                    if labels_list and i < len(labels_list)
                    else f"Item {i + 1}"
                ),
                "children": None,
            }
            for i in range(n)
        ]
        return (nodes, False)

    # Nested: list of dicts with "label" and "children"
    if isinstance(first, dict):
        nodes = []
        for i, item in enumerate(vals_list):
            if not isinstance(item, dict):
                continue
            children_raw = item.get("children")
            label = item.get("label", f"Item {i + 1}")
            if children_raw is not None:
                child_labels = None
                if labels and i < len(labels):
                    lab = labels[i]
                    if isinstance(lab, (list, tuple)):
                        child_labels = lab
                sub_nodes, _ = normalize_treemap_tree(children_raw, child_labels)
                value = sum(n["value"] for n in sub_nodes)
                nodes.append({"value": value, "label": label, "children": sub_nodes})
            else:
                value = float(item.get("value", 0))
                nodes.append({"value": value, "label": label, "children": None})
        return (nodes, any(n.get("children") for n in nodes))

    # Nested: list of lists [[100, 50], [30, 20]]
    if isinstance(first, (list, tuple)):
        labels_nested = list(labels) if labels else []
        top_labels = (
            labels_nested[0]
            if labels_nested and isinstance(labels_nested[0], (list, tuple))
            else labels_nested
        )
        child_labels_list = (
            labels_nested[1]
            if len(labels_nested) > 1 and isinstance(labels_nested[1], (list, tuple))
            else None
        )
        nodes = []
        for i, group in enumerate(vals_list):
            if not isinstance(group, (list, tuple)):
                continue
            sub_labels = (
                child_labels_list[i]
                if child_labels_list and i < len(child_labels_list)
                else None
            )
            sub_nodes, _ = normalize_treemap_tree(group, sub_labels)
            value = sum(n["value"] for n in sub_nodes)
            group_label = f"Group {i + 1}"
            if top_labels and i < len(top_labels) and isinstance(top_labels[i], str):
                group_label = top_labels[i]
            nodes.append({"value": value, "label": group_label, "children": sub_nodes})
        return (nodes, True)

    return ([], False)


def get_treemap_level(nodes: list[TreemapNode], path: list[int]) -> list[TreemapNode]:
    """Get the node list at the given path. path=[] returns nodes."""
    current: list[TreemapNode] = nodes
    for idx in path:
        if idx < 0 or idx >= len(current):
            return []
        current = current[idx].get("children") or []
    return current


def get_treemap_node_at_path(
    tree: list[TreemapNode], path: list[int]
) -> TreemapNode | None:
    """Get node at path. path=[0,1] -> tree[0].children[1]."""
    if not path:
        return None
    cur: list[TreemapNode] = tree
    for i in path[:-1]:
        if i < 0 or i >= len(cur):
            return None
        cur = cur[i].get("children") or []
    if path[-1] < 0 or path[-1] >= len(cur):
        return None
    return cur[path[-1]]


def get_path_styles(
    path: list[int],
    path_to_style: dict[tuple[int, ...], str],
) -> list[str]:
    """Get style for each node in path from path_to_style mapping."""
    if not path or not path_to_style:
        return []
    return [
        path_to_style[tuple(path[:i])]
        for i in range(1, len(path) + 1)
        if tuple(path[:i]) in path_to_style
    ]


def format_treemap_nested_path(
    tree: list[TreemapNode],
    path: list[int],
    total: float,
    value_display: "ValueDisplay",
    currency_symbol: str,
) -> tuple[list[tuple[str, str | None]], str]:
    """Build breadcrumb for nested treemap. Returns ([(label, style), ...], plain_text)."""
    if not tree or not path or total <= 0:
        return ([], "")
    from textual_plot.plot_widget import ValueDisplay

    segments: list[tuple[str, str | None]] = []
    plain_parts: list[str] = []
    current_nodes: list[TreemapNode] = tree
    parent_value = total
    for depth, idx in enumerate(path):
        if idx < 0 or idx >= len(current_nodes):
            break
        node = current_nodes[idx]
        value = node["value"]
        label = node.get("label", "?")
        pct_parent = 100 * value / parent_value if parent_value else 0
        pct_all = 100 * value / total if total else 0
        if value_display == ValueDisplay.CURRENCY:
            value_str = f"{currency_symbol}{value:,.2f}"
        else:
            value_str = f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"
        if depth == 0:
            segments.append((f"{label} ▪ {value_str} ▪ {pct_all:.1f}%", None))
            plain_parts.append(f"{label} ▪ {value_str} ▪ {pct_all:.1f}%")
        else:
            segments.append(
                (
                    f"{label} ▪ {value_str} ▪ {pct_parent:.1f}% ({pct_all:.1f}% All)",
                    None,
                )
            )
            plain_parts.append(
                f"{label} ▪ {value_str} ▪ {pct_parent:.1f}% ({pct_all:.1f}% All)"
            )
        current_nodes = node.get("children") or []
        parent_value = value
    return (segments, "  ▶  ".join(plain_parts))


def squarify_recursive(
    nodes: list[TreemapNode],
    x: float,
    y: float,
    dx: float,
    dy: float,
    aspect: float,
    path: list[int],
    base_styles: list[str],
    exclude_colors: list[tuple[float, float, float]],
    path_to_style: dict[tuple[int, ...], str] | None = None,
) -> list[dict]:
    """Recursively run squarify on a tree. Returns rect info for parents and leaves.

    Parents get distinctipy colors; children get distinctipy (excluding parent) tinted
    with parent's hue. At every nesting level: parents output first (draw underneath);
    children inset by TREEMAP_PARENT_PAD so parent shows as 1-char border.

    When path_to_style is provided, uses those styles for nodes instead of generating
    new ones (keeps colors consistent when zooming).
    """
    if not nodes or dx <= 0 or dy <= 0:
        return []
    out: list[dict] = []
    values = [n["value"] for n in nodes]
    normalized = squarify.normalize_sizes(values, dx, dy)
    rects = squarify.squarify(normalized, 0, 0, dx, dy)
    pad = TREEMAP_PARENT_PAD
    # Horizontal: same as bottom (2 chars) so parent border visible on left and right
    inset_x = float(pad) * 2 / max(0.01, aspect)
    inset_y_top = float(TREEMAP_PARENT_LABEL_TOP)
    inset_y_bottom = float(pad) * 2  # 2 rows at bottom so parent border is visible
    for i, rect in enumerate(rects):
        if i >= len(nodes):
            break
        node = nodes[i]
        rx, ry = rect["x"], rect["y"]
        rdx, rdy = rect["dx"], rect["dy"]
        child_path = path + [i]
        path_key = tuple(child_path)
        if path_to_style is not None and path_key in path_to_style:
            style = path_to_style[path_key]
        else:
            style = base_styles[i] if i < len(base_styles) else base_styles[0]
        if node.get("children"):
            parent_rgb = parse_style_to_rgb(style)
            n_children = len(node["children"])
            child_styles = []
            if path_to_style is not None:
                for j in range(n_children):
                    child_path_key = tuple(child_path + [j])
                    if child_path_key in path_to_style:
                        child_styles.append(path_to_style[child_path_key])
                    else:
                        child_styles.append(
                            base_styles[0] if base_styles else "rgb(128,128,128)"
                        )
            else:
                child_exclude = (
                    [parent_rgb] + exclude_colors
                    if parent_rgb
                    else list(exclude_colors)
                )
                child_exclude.extend(
                    [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)]
                )  # always exclude white/black
                child_colors = distinctipy.get_colors(
                    n_children,
                    exclude_colors=child_exclude,
                    pastel_factor=0.2,
                    rng=42,
                    colorblind_type="Deuteranomaly",
                )
                current_exclude = list(child_exclude)
                for c in child_colors:
                    tinted = tint_with_hue(c, parent_rgb) if parent_rgb else c
                    retries = 0
                    while (
                        parent_rgb
                        and rgb_too_close(tinted, parent_rgb)
                        and retries < 50
                    ):
                        current_exclude.append(c)
                        current_exclude.append(tinted)
                        c = distinctipy.distinct_color(
                            current_exclude,
                            pastel_factor=0.2,
                            rng=42,
                            colorblind_type="Deuteranomaly",
                        )
                        tinted = tint_with_hue(c, parent_rgb)
                        retries += 1
                    child_styles.append(rgb_style(tinted))
                    current_exclude.append(c)
                    current_exclude.append(tinted)
            parent_value = sum(n["value"] for n in node["children"])
            out.append(
                {
                    "x": x + rx,
                    "y": y + ry,
                    "dx": rdx,
                    "dy": rdy,
                    "node": node,
                    "path": child_path,
                    "style": style,
                    "selection_base": style,
                    "value": parent_value,
                    "label": node["label"],
                    "has_children": True,
                }
            )
            child_dx = max(0.1, rdx - 2 * inset_x)
            child_dy = max(0.1, rdy - inset_y_top - inset_y_bottom)
            child_x = x + rx + inset_x
            child_y = y + ry + inset_y_top
            child_exclude_list = (
                [parent_rgb] + exclude_colors if parent_rgb else exclude_colors
            )
            sub = squarify_recursive(
                node["children"],
                child_x,
                child_y,
                child_dx,
                child_dy,
                aspect,
                child_path,
                child_styles,
                child_exclude_list,
                path_to_style,
            )
            out.extend(sub)
        else:
            out.append(
                {
                    "x": x + rx,
                    "y": y + ry,
                    "dx": rdx,
                    "dy": rdy,
                    "node": node,
                    "path": child_path,
                    "style": style,
                    "selection_base": style,
                    "value": node["value"],
                    "label": node["label"],
                    "has_children": False,
                }
            )
    return out
