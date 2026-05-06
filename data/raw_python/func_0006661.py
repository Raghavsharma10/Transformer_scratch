def get_option_path(self, name, section=None, vars=None, expect=None):
        """Just like ``get_option`` but return a ``pathlib.Path`` object of
        the string.

        """
        val = self.get_option(name, section, vars, expect)
        return Path(val)