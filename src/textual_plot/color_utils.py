"""Color utilities for treemap and plot styling."""

from __future__ import annotations

import colorsys
import re

import distinctipy


def parse_style_to_rgb(style: str) -> tuple[float, float, float] | None:
    """Parse rgb(r,g,b) style string to (r,g,b) tuple in 0-1 range."""
    m = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style, re.I)
    if m:
        return (int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255)
    return None


def adjust_luminance(
    rgb: tuple[float, float, float], factor: float
) -> tuple[float, float, float]:
    """Adjust luminance of RGB (0-1) by factor. factor>1 lightens, factor<1 darkens."""
    h, lightness, s = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
    lightness = max(0, min(1, lightness * factor))
    return colorsys.hls_to_rgb(h, lightness, s)


def tint_with_hue(
    child_rgb: tuple[float, float, float],
    parent_rgb: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Apply parent's hue to child color; keep child's saturation and value for distinction."""
    ph, ps, pv = colorsys.rgb_to_hsv(parent_rgb[0], parent_rgb[1], parent_rgb[2])
    ch, cs, cv = colorsys.rgb_to_hsv(child_rgb[0], child_rgb[1], child_rgb[2])
    r, g, b = colorsys.hsv_to_rgb(ph, cs, cv)
    return (r, g, b)


def rgb_too_close(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """True if a and b are too similar (perceptual distance)."""
    return distinctipy.color_distance(a, b) < 0.02


def rgb_style(rgb: tuple[float, float, float]) -> str:
    """Convert (r,g,b) 0-1 to rgb(r,g,b) style string."""
    return f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"
