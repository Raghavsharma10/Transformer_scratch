def initinfo(self) -> Tuple[Union[float, int, bool], bool]:
        """The actual initial value of the given parameter.

        Some |Parameter| subclasses define another value for class
        attribute `INIT` than |None| to provide a default value.

        Let's define a parameter test class and prepare a function for
        initialising it and connecting the resulting instance to a
        |SubParameters| object:

        >>> from hydpy.core.parametertools import Parameter, SubParameters
        >>> class Test(Parameter):
        ...     NDIM = 0
        ...     TYPE = float
        ...     TIME = None
        ...     INIT = 2.0
        >>> class SubGroup(SubParameters):
        ...     CLASSES = (Test,)
        >>> def prepare():
        ...     subpars = SubGroup(None)
        ...     test = Test(subpars)
        ...     test.__hydpy__connect_variable2subgroup__()
        ...     return test

        By default, making use of the `INIT` attribute is disabled:

        >>> test = prepare()
        >>> test
        test(?)

        Enable it through setting |Options.usedefaultvalues| to |True|:

        >>> from hydpy import pub
        >>> pub.options.usedefaultvalues = True
        >>> test = prepare()
        >>> test
        test(2.0)

        When no `INIT` attribute is defined, enabling
        |Options.usedefaultvalues| has no effect, of course:

        >>> del Test.INIT
        >>> test = prepare()
        >>> test
        test(?)

        For time-dependent parameter values, the `INIT` attribute is assumed
        to be related to a |Parameterstep| of one day:

        >>> test.parameterstep = '2d'
        >>> test.simulationstep = '12h'
        >>> Test.INIT = 2.0
        >>> Test.TIME = True
        >>> test = prepare()
        >>> test
        test(4.0)
        >>> test.value
        1.0
        """
        init = self.INIT
        if (init is not None) and hydpy.pub.options.usedefaultvalues:
            with Parameter.parameterstep('1d'):
                return self.apply_timefactor(init), True
        return variabletools.TYPE2MISSINGVALUE[self.TYPE], False