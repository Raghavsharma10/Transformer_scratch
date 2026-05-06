def get_symbol_map(self):
        """
        If you need the symbol map, use this method.

        The symbol map is an array of string pairs mapping common tokens
        to X Keysym strings, such as "alt" to "Alt_L"

        :return: array of strings.
        """
        # todo: make sure we return a list of strings!
        sm = _libxdo.xdo_get_symbol_map()

        # Return value is like:
        # ['alt', 'Alt_L', ..., None, None, None, ...]
        # We want to return only values up to the first None.
        # todo: any better solution than this?
        i = 0
        ret = []
        while True:
            c = sm[i]
            if c is None:
                return ret
            ret.append(c)
            i += 1