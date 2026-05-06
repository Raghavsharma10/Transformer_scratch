def get_option_boolean(self, name, section=None, vars=None, expect=None):
        """Just like ``get_option`` but parse as a boolean (any case `true`).

        """
        val = self.get_option(name, section, vars, expect)
        val = val.lower() if val else 'false'
        return val == 'true'