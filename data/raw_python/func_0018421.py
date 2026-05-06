def set_size(self, pt=None, px=None):
        """
        Set the size of the font, in px or pt.

        The px method is a bit inacurate, there can be one or two px less, and max 4 for big numbers (like 503)
        but the size is never over-estimated. It makes almost the good value.
        """

        assert (pt, px) != (None, None)

        if pt is not None:
            self.__init__(pt, self.font_name)
        else:
            self.__init__(self.px_to_pt(px), self.font_name)