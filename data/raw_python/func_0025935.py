def substract(self, pt):
        """Return a Point instance as the displacement of two points."""
        if isinstance(pt, Point):
                return Point(pt.x - self.x, pt.y - self.y, pt.z - self.z)
        else:
            raise TypeError