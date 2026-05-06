def parse(value, pattern='{head}{padding}{tail} [{ranges}]'):
    '''Parse *value* into a :py:class:`~clique.collection.Collection`.

    Use *pattern* to extract information from *value*. It may make use of the
    following keys:

        * *head* - Common leading part of the collection.
        * *tail* - Common trailing part of the collection.
        * *padding* - Padding value in ``%0d`` format.
        * *range* - Total range in the form ``start-end``.
        * *ranges* - Comma separated ranges of indexes.
        * *holes* - Comma separated ranges of missing indexes.

    .. note::

        *holes* only makes sense if *range* or *ranges* is also present.

    '''
    # Construct regular expression for given pattern.
    expressions = {
        'head': '(?P<head>.*)',
        'tail': '(?P<tail>.*)',
        'padding': '%(?P<padding>\d*)d',
        'range': '(?P<range>\d+-\d+)?',
        'ranges': '(?P<ranges>[\d ,\-]+)?',
        'holes': '(?P<holes>[\d ,\-]+)'
    }

    pattern_regex = re.escape(pattern)
    for key, expression in expressions.items():
        pattern_regex = pattern_regex.replace(
            '\{{{0}\}}'.format(key),
            expression
        )
    pattern_regex = '^{0}$'.format(pattern_regex)

    # Match pattern against value and use results to construct collection.
    match = re.search(pattern_regex, value)
    if match is None:
        raise ValueError('Value did not match pattern.')

    groups = match.groupdict()
    if 'padding' in groups and groups['padding']:
        groups['padding'] = int(groups['padding'])
    else:
        groups['padding'] = 0

    # Create collection and then add indexes.
    collection = Collection(
        groups.get('head', ''),
        groups.get('tail', ''),
        groups['padding']
    )

    if groups.get('range', None) is not None:
        start, end = map(int, groups['range'].split('-'))
        collection.indexes.update(range(start, end + 1))

    if groups.get('ranges', None) is not None:
        parts = [part.strip() for part in groups['ranges'].split(',')]
        for part in parts:
            index_range = list(map(int, part.split('-', 2)))

            if len(index_range) > 1:
                # Index range.
                for index in range(index_range[0], index_range[1] + 1):
                    collection.indexes.add(index)
            else:
                # Single index.
                collection.indexes.add(index_range[0])

    if 'holes' in groups:
        parts = [part.strip() for part in groups['holes'].split(',')]
        for part in parts:
            index_range = map(int, part.split('-', 2))

            if len(index_range) > 1:
                # Index range.
                for index in range(index_range[0], index_range[1] + 1):
                    collection.indexes.remove(index)
            else:
                # Single index.
                collection.indexes.remove(index_range[0])

    return collection