def loadcurve(self, fsn: int) -> classes2.Curve:
        """Load a radial scattering curve"""
        return classes2.Curve.new_from_file(self.find_file(self._exposureclass + '_%05d.txt' % fsn))