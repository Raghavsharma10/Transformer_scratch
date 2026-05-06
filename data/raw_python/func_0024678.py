def best_prefix(self, system=None):
        """Optional parameter, `system`, allows you to prefer NIST or SI in
the results. By default, the current system is used (Bit/Byte default
to NIST).

Logic discussion/notes:

Base-case, does it need converting?

If the instance is less than one Byte, return the instance as a Bit
instance.

Else, begin by recording the unit system the instance is defined
by. This determines which steps (NIST_STEPS/SI_STEPS) we iterate over.

If the instance is not already a ``Byte`` instance, convert it to one.

NIST units step up by powers of 1024, SI units step up by powers of
1000.

Take integer value of the log(base=STEP_POWER) of the instance's byte
value. E.g.:

    >>> int(math.log(Gb(100).bytes, 1000))
    3

This will return a value >= 0. The following determines the 'best
prefix unit' for representation:

* result == 0, best represented as a Byte
* result >= len(SYSTEM_STEPS), best represented as an Exbi/Exabyte
* 0 < result < len(SYSTEM_STEPS), best represented as SYSTEM_PREFIXES[result-1]

        """

        # Use absolute value so we don't return Bit's for *everything*
        # less than Byte(1). From github issue #55
        if abs(self) < Byte(1):
            return Bit.from_other(self)
        else:
            if type(self) is Byte:  # pylint: disable=unidiomatic-typecheck
                _inst = self
            else:
                _inst = Byte.from_other(self)

        # Which table to consult? Was a preferred system provided?
        if system is None:
            # No preference. Use existing system
            if self.system == 'NIST':
                _STEPS = NIST_PREFIXES
                _BASE = 1024
            elif self.system == 'SI':
                _STEPS = SI_PREFIXES
                _BASE = 1000
            # Anything else would have raised by now
        else:
            # Preferred system provided.
            if system == NIST:
                _STEPS = NIST_PREFIXES
                _BASE = 1024
            elif system == SI:
                _STEPS = SI_PREFIXES
                _BASE = 1000
            else:
                raise ValueError("Invalid value given for 'system' parameter."
                                 " Must be one of NIST or SI")

        # Index of the string of the best prefix in the STEPS list
        _index = int(math.log(abs(_inst.bytes), _BASE))

        # Recall that the log() function returns >= 0. This doesn't
        # map to the STEPS list 1:1. That is to say, 0 is handled with
        # special care. So if the _index is 1, we actually want item 0
        # in the list.

        if _index == 0:
            # Already a Byte() type, so return it.
            return _inst
        elif _index >= len(_STEPS):
            # This is a really big number. Use the biggest prefix we've got
            _best_prefix = _STEPS[-1]
        elif 0 < _index < len(_STEPS):
            # There is an appropriate prefix unit to represent this
            _best_prefix = _STEPS[_index - 1]

        _conversion_method = getattr(
            self,
            'to_%sB' % _best_prefix)

        return _conversion_method()