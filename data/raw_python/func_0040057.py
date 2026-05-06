def GetAll(cls, attr, value, e=0.000001, sort_by="__name__"):
        """Get all nested Constant class that met ``klass.attr == value``.

        :param attr: attribute name.
        :param value: value.
        :param e: used for float value comparison.
        :param sort_by: nested class is ordered by <sort_by> attribute.

        .. versionadded:: 0.0.5
        """
        matched = list()
        for _, klass in cls.Subclasses(sort_by=sort_by):
            try:
                if klass.__dict__[attr] == approx(value, e):
                    matched.append(klass)
            except:  # pragma: no cover
                pass

        return matched