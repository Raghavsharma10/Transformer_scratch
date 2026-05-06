def colRowIsOnFgsPixel(self, col, row, padding=-50):
        """Is col row on a science pixel?

        #See Kepler Flight Segment User's Manual (SP0039-702) \S 5.4 (p88)

        Inputs:
        col, row (floats or ints)
        padding    If padding <0, pixel must be on silicon and this many
                   pixels away from the edge of the CCD to return True

        Returns:
        boolean
        """
        if col < 12. - padding  or col > 547 + padding:
            return False

        if row < 0 - padding or row > 527 + padding :
            return False
        return True