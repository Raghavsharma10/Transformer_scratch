def move(self, move):
        """Change the translation of this transform by the amount given.

        Parameters
        ----------
        move : array-like
            The values to be added to the current translation of the transform.
        """
        move = as_vec4(move, default=(0, 0, 0, 0))
        self.translate = self.translate + move