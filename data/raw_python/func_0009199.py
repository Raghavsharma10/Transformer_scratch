def matrix(self, angle):
        """Return rotation matrix in homogeneous coordinates."""
        _rot_mat = {
            'x': self._x_rot,
            'y': self._y_rot,
            'z': self._z_rot
        }
        return _rot_mat[self.axis](angle)