def alignment_a(self):
        """Computes the rotation matrix that aligns the unit cell with the
           Cartesian axes, starting with cell vector a.

           * a parallel to x
           * b in xy-plane with b_y positive
           * c with c_z positive
        """
        from molmod.transformations import Rotation
        new_x = self.matrix[:, 0].copy()
        new_x /= np.linalg.norm(new_x)
        new_z = np.cross(new_x, self.matrix[:, 1])
        new_z /= np.linalg.norm(new_z)
        new_y = np.cross(new_z, new_x)
        new_y /= np.linalg.norm(new_y)
        return Rotation(np.array([new_x, new_y, new_z]))