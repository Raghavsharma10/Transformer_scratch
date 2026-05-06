def swipe_by_coordinates(self, sx, sy, ex, ey, steps=10):
        """
        Swipe from (sx, sy) to (ex, ey) with *steps* .

        Example:
        | Swipe By Coordinates | 540 | 1340 | 940 | 1340 |     | # Swipe from (540, 1340) to (940, 100) with default steps 10 |
        | Swipe By Coordinates | 540 | 1340 | 940 | 1340 | 100 | # Swipe from (540, 1340) to (940, 100) with steps 100        |
        """
        self.device.swipe(sx, sy, ex, ey, steps)