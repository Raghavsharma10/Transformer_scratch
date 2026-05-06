def update_info(self, custom=None):
        """Updates the figure's suptitle.

        Calls self.info_string() unless custom is provided.

        Args:
            custom: Overwrite it with this string, unless None.
        """
        self.figure.suptitle(self.info_string() if custom is None else custom)