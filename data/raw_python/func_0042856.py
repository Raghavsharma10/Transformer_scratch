def typecheck(self, t):
        """Create a typecheck from some value ``t``.  This behaves differently
        depending on what ``t`` is.  It should take a value and return True if
        the typecheck passes, or False otherwise.  Override ``pre_validate``
        in a child class to do type coercion.

        * If ``t`` is a type, like basestring, int, float, *or* a tuple of base
          types, then a simple isinstance typecheck is returned.

        * If ``t`` is a list or tuple of instances, such as a tuple or list of
          integers or of strings, it's treated as the definition of an enum
          and a simple "in" check is returned.

        * If ``t`` is callable, ``t`` is assumed to be a valid typecheck.

        * If ``t`` is None, a typecheck that always passes is returned.

        If none of these conditions are met, a TypeError is raised.
        """
        if t is None:
            return lambda x: True

        def _isinstance(types, value):
            return isinstance(value, types)

        def _enum(values, value):
            return value in values

        if t.__class__ is type:
            return partial(_isinstance, t)
        elif isinstance(t, (tuple, list)):
            if all([x.__class__ is type for x in t]):
                return partial(_isinstance, t)
            return partial(_enum, t)
        elif callable(t):
            return t
        raise TypeError('%r is not a valid field type' % r)