def get_option_float(self, name, section=None, vars=None, expect=None):
        """Just like ``get_option`` but parse as a float."""
        val = self.get_option(name, section, vars, expect)
        if val:
            return float(val)