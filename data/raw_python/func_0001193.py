def by_interval_lookup(self, style_key, style_value):
        """Return a processor for an "interval" style value.

        Parameters
        ----------
        style_key : str
            A style key.
        style_value : dict
            A dictionary with an "interval" key whose value consists of a
            sequence of tuples where each tuple should have the form `(start,
            end, x)`, where start is the start of the interval (inclusive), end
            is the end of the interval, and x is either a style attribute (str)
            and a boolean flag indicating to use the style attribute named by
            `style_key`.

        Returns
        -------
        A function.
        """
        style_attr = style_key if self.style_types[style_key] is bool else None
        intervals = style_value["interval"]

        def proc(value, result):
            try:
                value = float(value)
            except TypeError:
                return result

            for start, end, lookup_value in intervals:
                if start is None:
                    start = float("-inf")
                if end is None:
                    end = float("inf")

                if start <= value < end:
                    if not lookup_value:
                        return result
                    return self.render(style_attr or lookup_value, result)
            return result
        return proc