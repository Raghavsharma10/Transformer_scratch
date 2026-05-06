def sub(cls, *mixins_and_dicts, **values):
        """Create and instantiate a sub-injector.

        Mixins and local value dicts can be passed in as arguments.  Local
        values can also be passed in as keyword arguments.
        """

        class SubInjector(cls):
            pass

        mixins = [ x for x in mixins_and_dicts if isinstance(x, type) ]
        if mixins:
            SubInjector.__bases__ = tuple(mixins) + SubInjector.__bases__

        dicts = [ x for x in mixins_and_dicts if not isinstance(x, type) ]
        for d in reversed(dicts):
            for k,v in d.items():
                if k not in values:
                    values[k] = v

        for k,v in values.items():
            SubInjector.value(k, v)

        return SubInjector()