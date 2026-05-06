def pixel_to_q(self, row: float, column: float):
        """Return the q coordinates of a given pixel.

        Inputs:
            row: float
                the row (vertical) coordinate of the pixel
            column: float
                the column (horizontal) coordinate of the pixel

        Coordinates are 0-based and calculated from the top left corner.
        """
        qrow = 4 * np.pi * np.sin(
            0.5 * np.arctan(
                (row - float(self.header.beamcentery)) *
                float(self.header.pixelsizey) /
                float(self.header.distance))) / float(self.header.wavelength)
        qcol = 4 * np.pi * np.sin(0.5 * np.arctan(
                (column - float(self.header.beamcenterx)) *
                float(self.header.pixelsizex) /
                float(self.header.distance))) / float(self.header.wavelength)
        return qrow, qcol