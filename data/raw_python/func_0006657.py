def get_option_list(self, name, section=None, vars=None,
                        expect=None, separator=','):
        """Just like ``get_option`` but parse as a list using ``split``.

        """
        val = self.get_option(name, section, vars, expect)
        return val.split(separator) if val else []