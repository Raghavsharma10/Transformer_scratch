def consider_member(name_member, member, module, class_=None):
        """Return |True| if the given member should be added to the
        substitutions. If not return |False|.

        Some examples based on the site-package |numpy|:

        >>> from hydpy.core.autodoctools import Substituter
        >>> import numpy

        A constant like |nan| should be added:

        >>> Substituter.consider_member(
        ...     'nan', numpy.nan, numpy)
        True

        Members with a prefixed underscore should not be added:

        >>> Substituter.consider_member(
        ...     '_NoValue', numpy._NoValue, numpy)
        False

        Members that are actually imported modules should not be added:

        >>> Substituter.consider_member(
        ...     'warnings', numpy.warnings, numpy)
        False

        Members that are actually defined in other modules should
        not be added:

        >>> numpy.Substituter = Substituter
        >>> Substituter.consider_member(
        ...     'Substituter', numpy.Substituter, numpy)
        False
        >>> del numpy.Substituter

        Members that are defined in submodules of a given package
        (either from the standard library or from site-packages)
        should be added...

        >>> Substituter.consider_member(
        ...     'clip', numpy.clip, numpy)
        True

        ...but not members defined in *HydPy* submodules:

        >>> import hydpy
        >>> Substituter.consider_member(
        ...     'Node', hydpy.Node, hydpy)
        False

        For descriptor instances (with method `__get__`) beeing members
        of classes should be added:

        >>> from hydpy.auxs import anntools
        >>> Substituter.consider_member(
        ...     'shape_neurons', anntools.ANN.shape_neurons,
        ...     anntools, anntools.ANN)
        True
        """
        if name_member.startswith('_'):
            return False
        if inspect.ismodule(member):
            return False
        real_module = getattr(member, '__module__', None)
        if not real_module:
            return True
        if real_module != module.__name__:
            if class_ and hasattr(member, '__get__'):
                return True
            if 'hydpy' in real_module:
                return False
            if module.__name__ not in real_module:
                return False
        return True