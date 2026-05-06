def post_from_style(self, column_style):
        """Yield post-format processors based on `column_style`.

        Parameters
        ----------
        column_style : dict
            A style where the top-level keys correspond to style attributes
            such as "bold" or "color".

        Returns
        -------
        A generator object.
        """
        flanks = Flanks()
        yield flanks.split_flanks

        fns = {"simple": self.by_key,
               "lookup": self.by_lookup,
               "re_lookup": self.by_re_lookup,
               "interval": self.by_interval_lookup}

        for key in self.style_types:
            if key not in column_style:
                continue

            vtype = value_type(column_style[key])
            fn = fns[vtype]
            args = [key, column_style[key]]
            if vtype == "re_lookup":
                args.append(sum(getattr(re, f)
                                for f in column_style.get("re_flags", [])))
            yield fn(*args)

        yield flanks.join_flanks