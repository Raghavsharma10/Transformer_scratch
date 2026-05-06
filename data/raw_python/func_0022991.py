def set_data(self, index, data):
        """Set the complete data for a single line strip.
        
        Parameters
        ----------
        index : int
            The index of the line strip to be replaced.
        data : array-like
            The data to assign to the selected line strip.
        """
        self._pos_tex[index, :] = data
        self.update()