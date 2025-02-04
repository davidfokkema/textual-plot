import enum

from textual.geometry import Size


class HiResMode(enum.Enum):
    HALFBLOCK = enum.auto()
    QUADRANT = enum.auto()
    BRAILLE = enum.auto()


hires_sizes = {
    HiResMode.HALFBLOCK: Size(1, 2),
    HiResMode.QUADRANT: Size(2, 2),
    HiResMode.BRAILLE: Size(2, 8),
}

pixels = {
    HiResMode.HALFBLOCK: {(0, 0): None, (1, 0): "▀", (0, 1): "▄", (1, 1): "█"},
    HiResMode.QUADRANT: {
        (0, 0, 0, 0): None,
        (0, 0, 0, 1): "▗",
        (0, 0, 1, 0): "▖",
        (0, 0, 1, 1): "▄",
        (0, 1, 0, 0): "▝",
        (0, 1, 0, 1): "▐",
        (0, 1, 1, 0): "▞",
        (0, 1, 1, 1): "▟",
        (1, 0, 0, 0): "▘",
        (1, 0, 0, 1): "▚",
        (1, 0, 1, 0): "▌",
        (1, 0, 1, 1): "▙",
        (1, 1, 0, 0): "▀",
        (1, 1, 0, 1): "▜",
        (1, 1, 1, 0): "▛",
        (1, 1, 1, 1): "█",
    },
}
