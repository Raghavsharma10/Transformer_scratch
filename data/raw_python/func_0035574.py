def is_blankspace(self, char):
        """
        Test if a character is a blankspace.

        Parameters
        ----------
        char : str
            The character to test.

        Returns
        -------
        ret : bool
            True if character is a blankspace, False otherwise.

        """
        if len(char) > 1:
            raise TypeError("Expected a char.")
        if char in self.blankspaces:
            return True
        return False