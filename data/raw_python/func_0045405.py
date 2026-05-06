def _set_fields(self, positional, options):
        """
        _parse_arguments
        """
        dictionary = {}

        if positional is not None and options is not None:
            self.positional = positional.copy()
            self.options = options.copy()
            dictionary = positional.copy()
            dictionary.update(options.copy())
            self.m_reprdict = {"positional": positional.copy(),
                               "options": options.copy()}

        for k, v in dictionary.items():
            if hasattr(v, "strip"):
                v = v.strip("'")
                v = v.strip('"')

            setattr(self, str(k), v)