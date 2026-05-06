def padded(self, padding):
        """Return a new Rect padded (smaller) by padding on all sides

        Parameters
        ----------
        padding : float
            The padding.

        Returns
        -------
        rect : instance of Rect
            The padded rectangle.
        """
        return Rect(pos=(self.pos[0]+padding, self.pos[1]+padding),
                    size=(self.size[0]-2*padding, self.size[1]-2*padding))