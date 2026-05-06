def _dist_to_trans(self, dist):
        """Convert mouse x, y movement into x, y, z translations"""
        rae = np.array([self.roll, self.azimuth, self.elevation]) * np.pi / 180
        sro, saz, sel = np.sin(rae)
        cro, caz, cel = np.cos(rae)
        dx = (+ dist[0] * (cro * caz + sro * sel * saz)
              + dist[1] * (sro * caz - cro * sel * saz))
        dy = (+ dist[0] * (cro * saz - sro * sel * caz)
              + dist[1] * (sro * saz + cro * sel * caz))
        dz = (- dist[0] * sro * cel + dist[1] * cro * cel)
        return dx, dy, dz