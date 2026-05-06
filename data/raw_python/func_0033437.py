def _parse_cli_facter_results(facter_results):
    '''Parse key value pairs printed with "=>" separators.
    YAML is preferred output scheme for facter.

    >>> list(_parse_cli_facter_results("""foo => bar
    ... baz => 1
    ... foo_bar => True"""))
    [('foo', 'bar'), ('baz', '1'), ('foo_bar', 'True')]
    >>> list(_parse_cli_facter_results("""foo => bar
    ... babababababababab
    ... baz => 2"""))
    [('foo', 'bar\nbabababababababab'), ('baz', '2')]
    >>> list(_parse_cli_facter_results("""3434"""))
    Traceback (most recent call last):
        ...
    ValueError: parse error


    Uses a generator interface:
    >>> _parse_cli_facter_results("foo => bar").next()
    ('foo', 'bar')
    '''
    last_key, last_value = None, []
    for line in filter(None, facter_results.splitlines()):
        res = line.split(six.u(" => "), 1)
        if len(res)==1:
            if not last_key:
                raise ValueError("parse error")
            else:
                last_value.append(res[0])
        else:
            if last_key:
                yield last_key, os.linesep.join(last_value)
            last_key, last_value = res[0], [res[1]]
    else:
        if last_key:
            yield last_key, os.linesep.join(last_value)