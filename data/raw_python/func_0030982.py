def _propertiesOrClustersForSequence(sequence, propertyNames, propertyValues,
                                     missingAAValue):
    """
    Extract amino acid property values or cluster numbers for a sequence.

    @param sequence: An C{AARead} (or a subclass) instance.
    @param propertyNames: An iterable of C{str} property names (each of which
        must be a key of a key in the C{propertyValues} C{dict}).
    @param propertyValues: A C{dict} in the form of C{PROPERTY_DETAILS} or
        C{PROPERTY_CLUSTERS} (see above).
    @param missingAAValue: A C{float} value to use for properties when an AA
        (e.g., 'X') is not known.
    @raise ValueError: If an unknown property is given in C{propertyNames}.
    @return: A C{dict} keyed by (lowercase) property name, with values that are
        C{list}s of the corresponding property value in C{propertyValues} in
        order of sequence position.
    """
    propertyNames = sorted(map(str.lower, set(propertyNames)))

    # Make sure all mentioned property names exist for at least one AA.
    knownProperties = set()
    for names in propertyValues.values():
        knownProperties.update(names)
    unknown = set(propertyNames) - knownProperties
    if unknown:
        raise ValueError(
            'Unknown propert%s: %s.' %
            ('y' if len(unknown) == 1 else 'ies', ', '.join(unknown)))

    aas = sequence.sequence.upper()
    result = {}

    for propertyName in propertyNames:
        result[propertyName] = values = []
        append = values.append
        for aa in aas:
            try:
                properties = propertyValues[aa]
            except KeyError:
                # No such AA.
                append(missingAAValue)
            else:
                append(properties[propertyName])

    return result