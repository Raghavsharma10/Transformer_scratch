def inverse(self):
        """ The inverse of this transform. 
        """
        if self._inverse is None:
            self._inverse = InverseTransform(self)
        return self._inverse