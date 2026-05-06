def circle(self, cx, cy, r, stroke=None, fill=None, stroke_width=1):
        """
        :param cx: Center X
        :param cy: Center Y
        :param r: Radius
        """
        self.put(' <circle cx="')
        self.put(str(cx))
        self.put('" cy="')
        self.put(str(cy))
        self.put('" r="')
        self.put(str(r))
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
        self.put('/>\n')