"""Request models and decimal-first validation.

Coordinates and radii MUST arrive as decimal strings so geometry is built
from the original decimal values.  Bare JSON numbers are rejected: a JSON
number is parsed as a binary float by every common stack, which would lose
the original decimal value before the service ever sees it.
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

_DECIMAL = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")

Side = Literal["left", "right"]
Sense = Literal["ccw", "cw"]


def _decimal_str(value, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            f"{label} must be a decimal string (e.g. \"12.50\"); JSON numbers "
            "are rejected to preserve the exact original value"
        )
    text = value.strip()
    if not _DECIMAL.match(text):
        raise ValueError(f"{label} is not a valid decimal literal: {value!r}")
    return text


PointInput = List[str]


class EntityInput(BaseModel):
    type: Literal["line", "arc"]
    start: PointInput
    end: PointInput
    center: Optional[PointInput] = None
    sense: Optional[Sense] = None
    full: bool = False
    side: Optional[Side] = None
    id: Optional[str] = None

    @field_validator("start", "end", "center", mode="before")
    @classmethod
    def _validate_point(cls, v, info):
        if v is None:
            return None
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"{info.field_name} must be [x, y] decimal strings")
        return [
            _decimal_str(v[0], f"{info.field_name}[0]"),
            _decimal_str(v[1], f"{info.field_name}[1]"),
        ]

    @field_validator("side", mode="before")
    @classmethod
    def _validate_side(cls, v):
        if v is None:
            return None
        if v not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return v


class AuditRequest(BaseModel):
    contour: List[EntityInput] = Field(min_length=1)
    tool_radius: str
    side: Side = "left"
    decimal_places: int = Field(default=12, ge=0, le=48)
    job: Optional[str] = None

    @field_validator("tool_radius", mode="before")
    @classmethod
    def _validate_radius(cls, v):
        return _decimal_str(v, "tool_radius")
