def format(self, *args, **kwargs):
        """
        Format the string representation of the editor.

        Args:
            inplace (bool): If True, overwrite editor's contents with formatted contents
        """
        inplace = kwargs.pop("inplace", False)
        if not inplace:
            return str(self).format(*args, **kwargs)
        self._lines = str(self).format(*args, **kwargs).splitlines()