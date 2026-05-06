def get_all(self, attr, value, e=0.000001,
                sort_by="__name__", reverse=False):
        """Get all nested Constant class that met ``klass.attr == value``.

        :param attr: attribute name.
        :param value: value.
        :param e: used for float value comparison.
        :param sort_by: nested class is ordered by <sort_by> attribute.

        .. versionchanged:: 0.0.5
        """
        matched = list()
        for _, klass in self.subclasses(sort_by, reverse):
            try:
                if getattr(klass, attr) == approx(value, e):
                    matched.append(klass)
            except:  # pragma: no cover
                pass

        return matched