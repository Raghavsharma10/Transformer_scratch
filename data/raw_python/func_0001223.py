def _find_local_signals(cls, signals,  namespace):
        """Add name info to every "local" (present in the body of this class)
        signal and add it to the mapping.  Also complete signal
        initialization as member of the class by injecting its name.
        """
        from . import Signal
        signaller = cls._external_signaller_and_handler
        for aname, avalue in namespace.items():
            if isinstance(avalue, Signal):
                if avalue.name:
                    aname = avalue.name
                else:
                    avalue.name = aname
                assert ((aname not in signals) or
                        (aname in signals and avalue is not signals[aname])), \
                        ("The same signal {name!r} was found "
                         "two times".format(name=aname))
                if signaller:
                    avalue.external_signaller = signaller
                signals[aname] = avalue