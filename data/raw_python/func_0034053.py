def send_target(self, x, y, cid=0):
        """
        Sets the target position of all cells.

        `x` and `y` are world coordinates. They can exceed the world border.

        For continuous movement, send a new target position
        before the old one is reached.

        In earlier versions of the game, it was possible to
        control each cell individually by specifying the cell's `cid`.

        Same as moving your mouse in the original client.
        """
        self.send_struct('<BiiI', 16, int(x), int(y), cid)