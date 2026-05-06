def GetFirst(cls, attr, value, e=0.000001, sort_by="__name__"):
        """Get the first nested Constant class that met ``klass.attr == value``.

        :param attr: attribute name.
        :param value: value.
        :param e: used for float value comparison.
        :param sort_by: nested class is ordered by <sort_by> attribute.

        .. versionadded:: 0.0.5
        """
        for _, klass in cls.Subclasses(sort_by=sort_by):
            try:
                if klass.__dict__[attr] == approx(value, e):
                    return klass
            except:
                pass

        return None