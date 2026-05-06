def drag_by_coordinates(self,sx, sy, ex, ey, steps=10):
        """
        Drag from (sx, sy) to (ex, ey) with steps

        See `Swipe By Coordinates` also.
        """
        self.device.drag(sx, sy, ex, ey, steps)