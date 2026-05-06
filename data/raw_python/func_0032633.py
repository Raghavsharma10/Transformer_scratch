def isPositiveMap(self):
        """Returns true if increasing ra increases pix in skyToPix()
        """
        x0, y0 = self.skyToPix(self.ra0_deg, self.dec0_deg)
        x1, y1 = self.skyToPix(self.ra0_deg + 1/3600., self.dec0_deg)

        if x1 > x0:
            return True
        return False