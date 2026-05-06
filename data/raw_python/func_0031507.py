def set(self, x, y, value):
        """
        Set the data at (x, y) to value.
        """
        xBase = int(x) * self.xScale
        yBase = int(y) * self.yScale
        for xOffset in range(self.xScale):
            for yOffset in range(self.yScale):
                self.data[yBase + yOffset, xBase + xOffset] = value