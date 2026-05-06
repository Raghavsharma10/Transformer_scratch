def get_role(member, cython=False):
        """Return the reStructuredText role `func`, `class`, or `const`
        best describing the given member.

        Some examples based on the site-package |numpy|.  |numpy.clip|
        is a function:

        >>> from hydpy.core.autodoctools import Substituter
        >>> import numpy
        >>> Substituter.get_role(numpy.clip)
        'func'

        |numpy.ndarray| is a class:

        >>> Substituter.get_role(numpy.ndarray)
        'class'

        |numpy.ndarray.clip| is a method, for which also the `function`
        role is returned:

        >>> Substituter.get_role(numpy.ndarray.clip)
        'func'

        For everything else the `constant` role is returned:

        >>> Substituter.get_role(numpy.nan)
        'const'

        When analysing cython extension modules, set the option `cython`
        flag to |True|.  |Double| is correctly identified as a class:

        >>> from hydpy.cythons import pointerutils
        >>> Substituter.get_role(pointerutils.Double, cython=True)
        'class'

        Only with the `cython` flag beeing |True|, for everything else
        the `function` text role is returned (doesn't make sense here,
        but the |numpy| module is not something defined in module
        |pointerutils| anyway):

        >>> Substituter.get_role(pointerutils.numpy, cython=True)
        'func'
        """
        if inspect.isroutine(member) or isinstance(member, numpy.ufunc):
            return 'func'
        elif inspect.isclass(member):
            return 'class'
        elif cython:
            return 'func'
        return 'const'