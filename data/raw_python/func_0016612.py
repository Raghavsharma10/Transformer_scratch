def copy_no_data(self):
        """
        Returns a copy of the object without any data.
        """
        return type(self)(
            [],
            order=list(self.header_modes),
            types=self.header_types.copy(),
            modes=self.header_modes.copy())