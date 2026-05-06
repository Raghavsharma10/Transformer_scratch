def diamondTabularFormatToDicts(filename, fieldNames=None):
    """
    Read DIAMOND tabular (--outfmt 6) output and convert lines to dictionaries.

    @param filename: Either a C{str} file name or an open file pointer.
    @param fieldNames: A C{list} or C{tuple} of C{str} DIAMOND field names.
        Run 'diamond -help' to see the full list. If C{None}, a default set of
        fields will be used, as compatible with convert-diamond-to-sam.py
    @raise ValueError: If a line of C{filename} does not have the expected
        number of TAB-separated fields (i.e., len(fieldNames)). Or if
        C{fieldNames} is empty or contains duplicates.
    @return: A generator that yields C{dict}s with keys that are the DIAMOND
        field names and values as converted by DIAMOND_FIELD_CONVERTER.
    """
    fieldNames = fieldNames or FIELDS.split()
    nFields = len(fieldNames)
    if not nFields:
        raise ValueError('fieldNames cannot be empty.')

    c = Counter(fieldNames)
    if c.most_common(1)[0][1] > 1:
        raise ValueError(
            'fieldNames contains duplicated names: %s.' %
            (', '.join(sorted(x[0] for x in c.most_common() if x[1] > 1))))

    def identity(x):
        return x

    convertFunc = DIAMOND_FIELD_CONVERTER.get

    with as_handle(filename) as fp:
        for count, line in enumerate(fp, start=1):
            result = {}
            line = line[:-1]
            values = line.split('\t')
            if len(values) != nFields:
                raise ValueError(
                    'Line %d of %s had %d field values (expected %d). '
                    'To provide input for this function, DIAMOND must be '
                    'called with "--outfmt 6 %s" (without the quotes). '
                    'The offending input line was %r.' %
                    (count,
                     (filename if isinstance(filename, six.string_types)
                      else 'input'),
                     len(values), nFields, FIELDS, line))
            for fieldName, value in zip(fieldNames, values):
                value = convertFunc(fieldName, identity)(value)
                result[fieldName] = value
            yield result