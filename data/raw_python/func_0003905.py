def from_properties(cls, angle, axis, invert, translation):
        """Initialize a transformation based on the properties"""
        rot = Rotation.from_properties(angle, axis, invert)
        return Complete(rot.r, translation)