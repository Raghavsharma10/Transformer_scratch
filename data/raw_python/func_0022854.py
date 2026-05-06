def _dist_to_trans(self, dist):
        """Convert mouse x, y movement into x, y, z translations"""
        rot, x, y, z = self._quaternion.get_axis_angle()
        tr = MatrixTransform()
        tr.rotate(180 * rot / np.pi, (x, y, z))
        dx, dz, dy = np.dot(tr.matrix[:3, :3], (dist[0], dist[1], 0.))
        return dx, dy, dz