def _rotate_tr(self):
        """Rotate the transformation matrix based on camera parameters"""
        rot, x, y, z = self._quaternion.get_axis_angle()
        up, forward, right = self._get_dim_vectors()
        self.transform.rotate(180 * rot / np.pi, (x, z, y))