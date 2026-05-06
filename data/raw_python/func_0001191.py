def by_lookup(self, style_key, style_value):
        """Return a processor that extracts the style from `mapping`.

        Parameters
        ----------
        style_key : str
            A style key.
        style_value : dict
            A dictionary with a "lookup" key whose value is a "mapping" style
            value that maps a field value to either a style attribute (str) and
            a boolean flag indicating to use the style attribute named by
            `style_key`.

        Returns
        -------
        A function.
        """
        style_attr = style_key if self.style_types[style_key] is bool else None
        mapping = style_value["lookup"]

        def proc(value, result):
            try:
                lookup_value = mapping[value]
            except (KeyError, TypeError):
                # ^ TypeError is included in case the user passes non-hashable
                # values.
                return result

            if not lookup_value:
                return result
            return self.render(style_attr or lookup_value, result)
        return proc