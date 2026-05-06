def translate(self, pos):
        """
        Translate the matrix

        The translation is applied *after* the transformations already present
        in the matrix.

        Parameters
        ----------
        pos : arrayndarray
            Position to translate by.
        """
        self.matrix = np.dot(self.matrix, transforms.translate(pos[0, :3]))