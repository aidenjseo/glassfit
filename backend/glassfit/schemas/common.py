"""Shared primitive schemas."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    import numpy as np

# Normalized MediaPipe coordinates: x,y in [0,1] relative to image width/height, z ~ x-scale.
Point3 = tuple[float, float, float]
# Pixel coordinates in the captured frame.
Point2 = tuple[float, float]


class LandmarkSet(BaseModel):
    """A canonical face-landmark set in normalized image coordinates."""

    points: list[Point3]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)

    @field_validator("points")
    @classmethod
    def _expected_count(cls, v: list[Point3]) -> list[Point3]:
        if len(v) not in (468, 478):
            raise ValueError(f"expected 468 or 478 landmarks, got {len(v)}")
        return v

    @property
    def has_iris(self) -> bool:
        return len(self.points) == 478

    @classmethod
    def from_array(cls, points: "np.ndarray", image_width: int, image_height: int) -> "LandmarkSet":
        """Build from an ``(N, 3)`` array of normalized coordinates."""
        return cls(
            points=[(float(p[0]), float(p[1]), float(p[2])) for p in points],
            image_width=image_width,
            image_height=image_height,
        )


class ImageSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class PerSide(BaseModel):
    """A per-side value. Sides are SUBJECT-anatomical (right = the subject's right / OD)."""

    right: float
    left: float
