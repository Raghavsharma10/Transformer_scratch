def un(source, wrapper=list, error_bad_lines=True):
    """Parse a text stream to TSV

    If the source is a string, it is converted to a line-iterable stream. If
    it is a file handle or other object, we assume that we can iterate over
    the lines in it.

    The result is a generator, and what it contains depends on whether the
    second argument is set and what it is set to.

    If the second argument is set to list, the default, then each element of
    the result is a list of strings. If it is set to a class generated with
    namedtuple(), then each element is an instance of this class, or None if
    there were too many or too few fields.

    Although newline separated input is preferred, carriage-return-newline is
    accepted on every platform.

    Since there is no definite order to the fields of a dict, there is no
    consistent way to format dicts for output. To avoid the asymmetry of a
    type that can be read but not written, plain dictionary parsing is
    omitted.
    """
    if isinstance(source, six.string_types):
        source = six.StringIO(source)

    # Prepare source lines for reading
    rows = parse_lines(source)

    # Get columns
    if is_namedtuple(wrapper):
        columns = wrapper._fields
        wrapper = wrapper._make
    else:
        columns = next(rows, None)
        if columns is not None:
            i, columns = columns
            yield wrapper(columns)

    # Get values
    for i, values in rows:
        if check_line_consistency(columns, values, i, error_bad_lines):
            yield wrapper(values)