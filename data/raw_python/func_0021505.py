def interpolate(self, other, t):
        """
        Create a new vertex between this vertex and `other` by linearly
        interpolating all properties using a parameter of `t`. Subclasses should
        override this to interpolate additional properties.
        """
        return Vertex(self.pos.lerp(other.pos, t), 
                          self.normal.lerp(other.normal, t))