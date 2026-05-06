def _rotate_tr(self):
        """Rotate the transformation matrix based on camera parameters"""
        up, forward, right = self._get_dim_vectors()
        self.transform.rotate(self.elevation, -right)
        self.transform.rotate(self.azimuth, up)