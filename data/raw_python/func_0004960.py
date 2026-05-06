def twotheta(self) -> ErrorValue:
        """Calculate the two-theta array"""
        row, column = np.ogrid[0:self.shape[0], 0:self.shape[1]]
        rho = (((self.header.beamcentery - row) * self.header.pixelsizey) ** 2 +
               ((self.header.beamcenterx - column) * self.header.pixelsizex) ** 2) ** 0.5
        assert isinstance(self.header.pixelsizex, ErrorValue)
        assert isinstance(self.header.pixelsizey, ErrorValue)
        assert isinstance(rho, ErrorValue)
        return (rho / self.header.distance).arctan()