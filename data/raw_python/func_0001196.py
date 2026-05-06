def render(self, style_attr, value):
        """Prepend terminal code for `key` to `value`.

        Parameters
        ----------
        style_attr : str
            A style attribute (e.g., "bold" or "blue").
        value : str
            The value to render.

        Returns
        -------
        The code for `key` (e.g., "\x1b[1m" for bold) plus the
        original value.
        """
        if not value.strip():
            # We've got an empty string.  Don't bother adding any
            # codes.
            return value
        return six.text_type(getattr(self.term, style_attr)) + value