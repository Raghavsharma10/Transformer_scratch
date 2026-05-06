def about_axis(cls, center, angle, axis, invert=False):
        """Create transformation that represents a rotation about an axis

           Arguments:
            | ``center``  --  Point on the axis
            | ``angle``  --  Rotation angle
            | ``axis``  --  Rotation axis
            | ``invert``  --  When True, an inversion rotation is constructed
                              [default=False]
        """
        return Translation(center) * \
               Rotation.from_properties(angle, axis, invert) * \
               Translation(-center)