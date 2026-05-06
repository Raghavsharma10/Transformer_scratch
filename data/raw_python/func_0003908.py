def properties(self):
        """Transformation properties: angle, axis, invert, translation"""
        rot = Rotation(self.r)
        angle, axis, invert = rot.properties
        return angle, axis, invert, self.t