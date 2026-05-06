def revert_timefactor(cls, values):
        """The inverse version of method |Parameter.apply_timefactor|.

        See the explanations on method Parameter.apply_timefactor| to
        understand the following examples:

        .. testsetup::

            >>> from hydpy import pub
            >>> del pub.timegrids

        >>> from hydpy.core.parametertools import Parameter
        >>> class Par(Parameter):
        ...     TIME = None
        >>> Par.parameterstep = '1d'
        >>> Par.simulationstep = '6h'
        >>> Par.revert_timefactor(4.0)
        4.0

        >>> Par.TIME = True
        >>> Par.revert_timefactor(4.0)
        16.0

        >>> Par.TIME = False
        >>> Par.revert_timefactor(4.0)
        1.0
        """
        if cls.TIME is True:
            return values / cls.get_timefactor()
        if cls.TIME is False:
            return values * cls.get_timefactor()
        return values