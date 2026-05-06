def arc(self, cx, cy, r, start_radians, end_radians, fill=None):
        """
        NOTE: This will leave gaps between adjacent segments. You can fix this by disabling anti-aliasing using
              shape-rendering="crispEdges" on the path tag

              If you want to add a stroke to this, it's advisable to create a second path that just strokes the
              outside of the circle

        :param cx: Center X
        :param cy: Center Y
        :param r: Radius
        :param start_radians: Start of arc in radians, clockwise from vertical
        :param end_radians: End of arc in radians, clockwise from vertical
        """

        # This is tricky - we have to use a path, and for that we need to know the start and end points of the arc
        # in cartesian coordinates

        start = self._polar_to_cartesian(cx, cy, r, start_radians)
        end = self._polar_to_cartesian(cx, cy, r, end_radians)

        total_radians = start_radians - end_radians
        if total_radians < 0:
            total_radians *= -1
        large_arc_flag = 1 if total_radians > math.pi else 0

        path = SvgPathGenerator()
        path.move_to(end[0], end[1])
        path.line_to(cx, cy)
        path.line_to(start[0], start[1])
        path.arc_to(r, r, 0, large_arc_flag, 0, end[0], end[1])

        self.put(' <path d="')
        self.put(path.get_d())
        self.put('"')
        if fill:
            self.put(' fill="')
            self.put(fill)
            self.put('"')
        self.put('/>\n')