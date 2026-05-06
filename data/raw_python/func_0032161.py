def colRowIsOnSciencePixel(self, col, row, padding=DEFAULT_PADDING):
        """Is col row on a science pixel?

        Ranges taken from Fig 25 or Instrument Handbook (p50)

        Padding allows for the fact that distortion means the
        results from getColRowWithinChannel can be off by a bit.
        Setting padding > 0 means that objects that are computed
        to lie a small amount off silicon will return True.

        To be conservative, set padding to negative
        """
        if col < 12. - padding or col > 1111 + padding:
            return False

        if row < 20 - padding or row > 1043 + padding:
            return False
        return True