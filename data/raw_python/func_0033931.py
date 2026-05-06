def setOptions(self, glob=False, **kwargs):
        """Set option(s).

        :glob: If True, stores specified options globally.
        :kwargs: Dictionary of options and values to set.

        """
        if glob:
            self.globalOptions.update(kwargs)
        else:
            self.options.update(kwargs)