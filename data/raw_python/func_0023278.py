def flipped(self, x=False, y=True):
        """Return a Rect with the same bounds but with axes inverted

        Parameters
        ----------
        x : bool
            Flip the X axis.
        y : bool
            Flip the Y axis.

        Returns
        -------
        rect : instance of Rect
            The flipped rectangle.
        """
        pos = list(self.pos)
        size = list(self.size)
        for i, flip in enumerate((x, y)):
            if flip:
                pos[i] += size[i]
                size[i] *= -1
        return Rect(pos, size)