def by_re_lookup(self, style_key, style_value, re_flags=0):
        """Return a processor for a "re_lookup" style value.

        Parameters
        ----------
        style_key : str
            A style key.
        style_value : dict
            A dictionary with a "re_lookup" style value that consists of a
            sequence of items where each item should have the form `(regexp,
            x)`, where regexp is a regular expression to match against the
            field value and x is either a style attribute (str) and a boolean
            flag indicating to use the style attribute named by `style_key`.
        re_flags : int
            Passed through as flags argument to re.compile.

        Returns
        -------
        A function.
        """
        style_attr = style_key if self.style_types[style_key] is bool else None
        regexps = [(re.compile(r, flags=re_flags), v)
                   for r, v in style_value["re_lookup"]]

        def proc(value, result):
            if not isinstance(value, six.string_types):
                return result
            for r, lookup_value in regexps:
                if r.search(value):
                    if not lookup_value:
                        return result
                    return self.render(style_attr or lookup_value, result)
            return result
        return proc