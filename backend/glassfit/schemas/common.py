"""Shared primitive schemas."""

from pydantic import BaseModel, Field, field_validator

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


class ImageSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class PerSide(BaseModel):
    """A per-side value. Sides are SUBJECT-anatomical (right = the subject's right / OD)."""

    right: float
    left: float
