def pan(self, *pan):
        """Pan the view.

        Parameters
        ----------
        *pan : length-2 sequence
            The distance to pan the view, in the coordinate system of the
            scene.
        """
        if len(pan) == 1:
            pan = pan[0]
        self.rect = self.rect + pan