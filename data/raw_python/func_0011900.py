def from_scale(scale_w, scale_h=None):
        """Creates a padding by the remaining space after scaling the content.

        E.g. Padding.from_scale(0.5) would produce Padding(0.25, 0.25, 0.25, 0.25) and
        Padding.from_scale(0.5, 1) would produce Padding(0.25, 0.25, 0, 0)
        because the content would not be scaled (since scale_h=1) and therefore
        there would be no vertical padding.
        
        If scale_h is not specified scale_h=scale_w is used as default

        :param scale_w: horizontal scaling factors
        :type scale_w: float
        :param scale_h: vertical scaling factor
        :type scale_h: float
        """
        if not scale_h: scale_h = scale_w
        w_padding = [(1 - scale_w) * 0.5] * 2
        h_padding = [(1 - scale_h) * 0.5] * 2
        return Padding(*w_padding, *h_padding)