def _compose(self, name, attributes):
        """Construct a style taking `attributes` from the column styles.

        Parameters
        ----------
        name : str
            Name of main style (e.g., "header_").
        attributes : set of str
            Adopt these elements from the column styles.

        Returns
        -------
        The composite style for `name`.
        """
        name_style = _safe_get(self.init_style, name, elements.default(name))
        if self.init_style is not None and name_style is not None:
            result = {}
            for col in self.columns:
                cstyle = {k: v for k, v in self.style[col].items()
                          if k in attributes}
                result[col] = dict(cstyle, **name_style)
            return result