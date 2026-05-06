def get_option_int(self, name, section=None, vars=None, expect=None):
        """Just like ``get_option`` but parse as an integer."""
        val = self.get_option(name, section, vars, expect)
        if val:
            return int(val)