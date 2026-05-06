def by_key(self, style_key, style_value):
        """Return a processor for a "simple" style value.

        Parameters
        ----------
        style_key : str
            A style key.
        style_value : bool or str
            A "simple" style value that is either a style attribute (str) and a
            boolean flag indicating to use the style attribute named by
            `style_key`.

        Returns
        -------
        A function.
        """
        if self.style_types[style_key] is bool:
            style_attr = style_key
        else:
            style_attr = style_value

        def proc(_, result):
            return self.render(style_attr, result)
        return proc