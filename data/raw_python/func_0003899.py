def from_properties(cls, angle, axis, invert):
        """Initialize a rotation based on the properties"""
        norm = np.linalg.norm(axis)
        if norm > 0:
            x = axis[0] / norm
            y = axis[1] / norm
            z = axis[2] / norm
            c = np.cos(angle)
            s = np.sin(angle)
            r = (1-2*invert) * np.array([
                [x*x*(1-c)+c  , x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                [x*y*(1-c)+z*s, y*y*(1-c)+c  , y*z*(1-c)-x*s],
                [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c  ]
            ])
        else:
            r = np.identity(3) * (1-2*invert)
        return cls(r)