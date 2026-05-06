def range_to_numeric(ranges):
    """Converts a sequence of string ranges to a sequence of floats.

    E.g.::

        >>> range_to_numeric(['1 uV', '2 mV', '1 V'])
        [1E-6, 0.002, 1.0]

    """
    values, units = zip(*(r.split() for r in ranges))
    # Detect common unit.
    unit = os.path.commonprefix([u[::-1] for u in units])

    # Strip unit to get just the SI prefix.
    prefixes = (u[:-len(unit)] for u in units)

    # Convert string value and scale with prefix.
    values = [float(v) * SI_PREFIX[p] for v, p in zip(values, prefixes)]
    return values