def unit(self):
        """Return a Vector instance of the unit vector"""
        return Vector(
            (self.x / self.magnitude()),
            (self.y / self.magnitude()),
            (self.z / self.magnitude())
        )