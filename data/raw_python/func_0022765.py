def get_matrix(self):
        """ Create a 4x4 homography matrix that represents the rotation
        of the quaternion.
        """
        # Init matrix (remember, a matrix, not an array)
        a = np.zeros((4, 4), dtype=np.float32)
        w, x, y, z = self.w, self.x, self.y, self.z
        # First row
        a[0, 0] = - 2.0 * (y * y + z * z) + 1.0
        a[1, 0] = + 2.0 * (x * y + z * w)
        a[2, 0] = + 2.0 * (x * z - y * w)
        a[3, 0] = 0.0
        # Second row
        a[0, 1] = + 2.0 * (x * y - z * w)
        a[1, 1] = - 2.0 * (x * x + z * z) + 1.0
        a[2, 1] = + 2.0 * (z * y + x * w)
        a[3, 1] = 0.0
        # Third row
        a[0, 2] = + 2.0 * (x * z + y * w)
        a[1, 2] = + 2.0 * (y * z - x * w)
        a[2, 2] = - 2.0 * (x * x + y * y) + 1.0
        a[3, 2] = 0.0
        # Fourth row
        a[0, 3] = 0.0
        a[1, 3] = 0.0
        a[2, 3] = 0.0
        a[3, 3] = 1.0
        return a