def clone(self, default):
        """
        Make a copy of this parameter, supplying a different default.

        @type default: C{unicode} or C{NoneType}
        @param default: A value which will be initially presented in the view
        as the value for this parameter, or C{None} if no such value is to be
        presented.

        @rtype: L{Parameter}
        """
        return self.__class__(
            self.name,
            self.type,
            self.coercer,
            self.label,
            self.description,
            default,
            self.viewFactory)