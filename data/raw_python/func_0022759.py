def _normalize(self):
        """ Make the quaternion unit length.
        """
        # Get length
        L = self.norm()
        if not L:
            raise ValueError('Quaternion cannot have 0-length.')
        # Correct
        self.w /= L
        self.x /= L
        self.y /= L
        self.z /= L