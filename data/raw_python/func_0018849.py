def average_values(self, *args, **kwargs) -> float:
        """Average the actual values of the |Variable| object.

        For 0-dimensional |Variable| objects, the result of method
        |Variable.average_values| equals |Variable.value|.  The
        following example shows this for the sloppily defined class
        `SoilMoisture`:

        >>> from hydpy.core.variabletools import Variable
        >>> class SoilMoisture(Variable):
        ...     NDIM = 0
        ...     TYPE = float
        ...     refweigths = None
        ...     availablemasks = None
        ...     __hydpy__connect_variable2subgroup__ = None
        ...     initinfo = None
        >>> sm = SoilMoisture(None)
        >>> sm.value = 200.0
        >>> sm.average_values()
        200.0

        When the dimensionality of this class is increased to one,
        applying method |Variable.average_values| results in the
        following error:

        >>> SoilMoisture.NDIM = 1
        >>> import numpy
        >>> SoilMoisture.shape = (3,)
        >>> SoilMoisture.value = numpy.array([200.0, 400.0, 500.0])
        >>> sm.average_values()
        Traceback (most recent call last):
        ...
        AttributeError: While trying to calculate the mean value \
of variable `soilmoisture`, the following error occurred: Variable \
`soilmoisture` does not define any weighting coefficients.

        So model developers have to define another (in this case
        1-dimensional) |Variable| subclass (usually a |Parameter|
        subclass), and make the relevant object available via property
        |Variable.refweights|:

        >>> class Area(Variable):
        ...     NDIM = 1
        ...     shape = (3,)
        ...     value = numpy.array([1.0, 1.0, 2.0])
        ...     __hydpy__connect_variable2subgroup__ = None
        ...     initinfo = None
        >>> area = Area(None)
        >>> SoilMoisture.refweights = property(lambda self: area)
        >>> sm.average_values()
        400.0

        In the examples above, all single entries of `values` are relevant,
        which is the default case.  However, subclasses of |Variable| can
        define an alternative mask, allowing to make some entries
        irrelevant. Assume for example, that our `SoilMoisture` object
        contains three single values, each one associated with a specific
        hydrological response unit (hru).  To indicate that soil moisture
        is undefined for the third unit, (maybe because it is a water area),
        we set the third entry of the verification mask to |False|:

        >>> from hydpy.core.masktools import DefaultMask
        >>> class Soil(DefaultMask):
        ...     @classmethod
        ...     def new(cls, variable, **kwargs):
        ...         return cls.array2mask([True, True, False])
        >>> SoilMoisture.mask = Soil()
        >>> sm.average_values()
        300.0

        Alternatively, method |Variable.average_values| accepts additional
        masking information as positional or keyword arguments.  Therefore,
        the corresponding model must implement some alternative masks,
        which are provided by property |Variable.availablemasks|.
        We mock this property with a new |Masks| object, handling one
        mask for flat soils (only the first hru), one mask for deep soils
        (only the second hru), and one mask for water areas (only the
        third hru):

        >>> class FlatSoil(DefaultMask):
        ...     @classmethod
        ...     def new(cls, variable, **kwargs):
        ...         return cls.array2mask([True, False, False])
        >>> class DeepSoil(DefaultMask):
        ...     @classmethod
        ...     def new(cls, variable, **kwargs):
        ...         return cls.array2mask([False, True, False])
        >>> class Water(DefaultMask):
        ...     @classmethod
        ...     def new(cls, variable, **kwargs):
        ...         return cls.array2mask([False, False, True])
        >>> from hydpy.core import masktools
        >>> class Masks(masktools.Masks):
        ...     CLASSES = (FlatSoil,
        ...                DeepSoil,
        ...                Water)
        >>> SoilMoisture.availablemasks = Masks(None)

        One can pass either the mask classes themselves or their names:

        >>> sm.average_values(sm.availablemasks.flatsoil)
        200.0
        >>> sm.average_values('deepsoil')
        400.0

        Both variants can be combined:

        >>> sm.average_values(sm.availablemasks.deepsoil, 'flatsoil')
        300.0

        The following error happens if the general mask of the variable
        does not contain the given masks:

        >>> sm.average_values('flatsoil', 'water')
        Traceback (most recent call last):
        ...
        ValueError: While trying to calculate the mean value of variable \
`soilmoisture`, the following error occurred: Based on the arguments \
`('flatsoil', 'water')` and `{}` the mask `CustomMask([ True, False,  True])` \
has been determined, which is not a submask of `Soil([ True,  True, False])`.

        Applying masks with custom options is also supported.  One can change
        the behaviour of the following mask via the argument `complete`:

        >>> class AllOrNothing(DefaultMask):
        ...     @classmethod
        ...     def new(cls, variable, complete):
        ...         if complete:
        ...             bools = [True, True, True]
        ...         else:
        ...             bools = [False, False, False]
        ...         return cls.array2mask(bools)
        >>> class Masks(Masks):
        ...     CLASSES = (FlatSoil,
        ...                DeepSoil,
        ...                Water,
        ...                AllOrNothing)
        >>> SoilMoisture.availablemasks = Masks(None)

        Again, one can apply the mask class directly (but note that one
        has to pass the relevant variable as the first argument.):

        >>> sm.average_values(   # doctest: +ELLIPSIS
        ...     sm.availablemasks.allornothing(sm, complete=True))
        Traceback (most recent call last):
        ...
        ValueError: While trying to...

        Alternatively, one can pass the mask name as a keyword and pack
        the mask's options into a |dict| object:

        >>> sm.average_values(allornothing={'complete': False})
        nan

        You can combine all variants explained above:

        >>> sm.average_values(
        ...     'deepsoil', flatsoil={}, allornothing={'complete': False})
        300.0
        """
        try:
            if not self.NDIM:
                return self.value
            mask = self.get_submask(*args, **kwargs)
            if numpy.any(mask):
                weights = self.refweights[mask]
                return numpy.sum(weights*self[mask])/numpy.sum(weights)
            return numpy.nan
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to calculate the mean value of variable '
                f'{objecttools.devicephrase(self)}')