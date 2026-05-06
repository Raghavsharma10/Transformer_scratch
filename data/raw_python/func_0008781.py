def walk_perimeter(self, startx, starty):
        """
        Starting at a point on the perimeter of a region, 'walk' the perimeter to return
        to the starting point. Record the path taken.

        Parameters
        ----------
        startx, starty : int
            The starting location. Assumed to be on the perimeter of a region.

        Returns
        -------
        perimeter : list
            A list of pixel coordinates [ [x1,y1], ...] that constitute the perimeter of the region.
        """
        # checks
        startx = max(startx, 0)
        startx = min(startx, self.xsize)
        starty = max(starty, 0)
        starty = min(starty, self.ysize)

        points = []

        x, y = startx, starty

        while True:
            self.step(x, y)
            if 0 <= x <= self.xsize and 0 <= y <= self.ysize:
                points.append((x, y))
            if self.next == self.UP:
                y -= 1
            elif self.next == self.LEFT:
                x -= 1
            elif self.next == self.DOWN:
                y += 1
            elif self.next == self.RIGHT:
                x += 1
            # stop if we meet some kind of error
            elif self.next == self.NOWHERE:
                break
            # stop when we return to the starting location
            if x == startx and y == starty:
                break
        return points