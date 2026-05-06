def _parse_url(url, fully_qualified=False):
    """Parse the given charm or bundle URL, provided as a string.

    Return a tuple containing the entity reference fragments: schema, user,
    series, name and revision.
    Each fragment is a string except revision (int).

    Raise a ValueError with a descriptive message if the given URL is not
    valid. If fully_qualified is True, the URL must include the schema, series
    and revision, otherwise a ValueError is raised.
    """
    # Retrieve the schema.
    try:
        schema, remaining = url.split(':', 1)
    except ValueError:
        if fully_qualified:
            msg = 'URL has no schema: {}'.format(url)
            raise ValueError(msg.encode('utf-8'))
        schema = 'cs'
        remaining = url
    if schema not in ('cs', 'local'):
        msg = 'URL has invalid schema: {}'.format(schema)
        raise ValueError(msg.encode('utf-8'))
    # Retrieve and validate the optional user.
    parts = remaining.split('/')
    part = parts.pop(0)
    user = ''
    if part.startswith('~'):
        user = part[1:]
        if not valid_user(user):
            msg = 'URL has invalid user name: {}'.format(user)
            raise ValueError(msg.encode('utf-8'))
        if schema == 'local':
            msg = 'local entity URL with user name: {}'.format(url)
            raise ValueError(msg.encode('utf-8'))
        if not parts:
            msg = 'URL has invalid form: {}'.format(url)
            raise ValueError(msg.encode('utf-8'))
        part = parts.pop(0)
    # Retrieve and validate the series.
    series = ''
    if parts:
        series = part
        if not valid_series(series):
            msg = 'URL has invalid series: {}'.format(series)
            raise ValueError(msg.encode('utf-8'))
        part = parts.pop(0)
    elif fully_qualified:
        msg = 'URL has invalid form: {}'.format(url)
        raise ValueError(msg.encode('utf-8'))
    # Retrieve and validate name and revision.
    if parts:
        msg = 'URL has invalid form: {}'.format(url)
        raise ValueError(msg.encode('utf-8'))
    try:
        name, revision = part.rsplit('-', 1)
    except ValueError:
        if fully_qualified:
            msg = 'URL has no revision: {}'.format(url)
            raise ValueError(msg.encode('utf-8'))
        name, revision = part, None
    if revision is not None:
        try:
            revision = int(revision)
        except ValueError:
            if fully_qualified:
                msg = 'URL has invalid revision: {}'.format(revision)
                raise ValueError(msg.encode('utf-8'))
            name, revision = name + '-' + revision, None
    if not valid_name(name):
        msg = 'URL has invalid name: {}'.format(name)
        raise ValueError(msg.encode('utf-8'))
    return schema, user, series, name, revision