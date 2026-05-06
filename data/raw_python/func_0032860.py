def _toComparableValue(self, value):
        """
        Trivial wrapper which takes into account the possibility that our sort
        column might not have defined the C{toComparableValue} method.

        This can probably serve as a good generic template for some
        infrastructure to deal with arbitrarily-potentially-missing methods
        from certain versions of interfaces, but we didn't take it any further
        than it needed to go for this system's fairly meagre requirements.
        *Please* feel free to refactor upwards as necessary.
        """
        if hasattr(self.currentSortColumn, 'toComparableValue'):
            return self.currentSortColumn.toComparableValue(value)
        # Retrieve the location of the class's definition so that we can alert
        # the user as to where they need to insert their implementation.
        classDef = self.currentSortColumn.__class__
        filename = inspect.getsourcefile(classDef)
        lineno = inspect.findsource(classDef)[1]
        warnings.warn_explicit(
            "IColumn implementor " + qual(self.currentSortColumn.__class__) + " "
            "does not implement method toComparableValue.  This is required since "
            "Mantissa 0.6.6.",
            DeprecationWarning, filename, lineno)
        return value