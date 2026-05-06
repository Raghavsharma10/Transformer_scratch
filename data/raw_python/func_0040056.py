def get_first(self, attr, value, e=0.000001,
                  sort_by="__name__", reverse=False):
        """Get the first nested Constant class that met ``klass.attr == value``.

        :param attr: attribute name.
        :param value: value.
        :param e: used for float value comparison.
        :param sort_by: nested class is ordered by <sort_by> attribute.

        .. versionchanged:: 0.0.5
        """
        for _, klass in self.subclasses(sort_by, reverse):
            try:
                if getattr(klass, attr) == approx(value, e):
                    return klass
            except:
                pass

        return None