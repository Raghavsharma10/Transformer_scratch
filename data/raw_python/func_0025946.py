def from_points(cls, point1, point2):
        """Return a Vector instance from two given points."""
        if isinstance(point1, Point) and isinstance(point2, Point):
            displacement = point1.substract(point2)
            return cls(displacement.x, displacement.y, displacement.z)
        raise TypeError