def polygon(self, points, stroke=None, fill=None, stroke_width=1, disable_anti_aliasing=False):
        """
        :param points: List of points
        """
        self.put(' <polygon points="')
        self.put(' '.join(['%s,%s' % p for p in points]))
        self.put('" stroke-width="')
        self.put(str(stroke_width))
        self.put('"')
        if fill:
            self.put(' fill="')
            self.put(fill)
            self.put('"')
        if stroke:
            self.put(' stroke="')
            self.put(stroke)
            self.put('"')
        if disable_anti_aliasing:
            self.put(' shape-rendering="crispEdges"')
        self.put('/>\n')