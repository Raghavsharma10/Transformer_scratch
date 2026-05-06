def normalize_dsn_or_dict(d):
    """Clean up a database DSN, or dict version of a DSN, returning both the cleaned DSN and dict version"""
    if isinstance(d, dict):

        try:
            # Convert from an AttrDict to a real dict
            d = d.to_dict()
        except AttributeError:
            pass  # Already a real dict

        config = d
        dsn = None

    elif isinstance(d, string_types):
        config = None
        dsn = d

    else:
        raise ConfigurationError("Can't deal with database config '{}' type '{}' ".format(d, type(d)))

    if dsn:

        if dsn.startswith('sqlite') or dsn.startswith('spatialite'):
            driver, path = dsn.split(':', 1)

            slashes, path = path[:2], path[2:]

            if slashes != '//':
                raise ConfigurationError("Sqlite DSNs must start with at least 2 slashes")

            if len(path) == 1 and path[0] == '/':
                raise ConfigurationError("Sqlite DSNs can't have only 3 slashes in path")

            if len(path) > 1 and path[0] != '/':
                raise ConfigurationError("Sqlite DSNs with a path must have 3 or 4 slashes.")

            path = path[1:]

            config = dict(
                server=None,
                username=None,
                password=None,
                driver=driver,
                dbname=path
            )
        else:

            d = parse_url_to_dict(dsn)

            config = dict(
                server=d['hostname'],
                dbname=d['path'].strip('/'),
                driver=d['scheme'],
                password=d.get('password', None),
                username=d.get('username', None)
            )
    else:
        up = d.get('username', '') or ''

        if d.get('password'):

            up += ':' + d.get('password', '')

        if up:
            up += '@'

        if up and not d.get('server'):
            raise ConfigurationError("Can't construct a DSN with a username or password without a hostname")

        host_part = up + d.get('server', '') if d.get('server') else ''

        if d.get('dbname', False):
            path_part = '/' + d.get('dbname')

            # if d['driver'] in ('sqlite3', 'sqlite', 'spatialite'):
            #     path_part = '/' + path_part

        else:
            path_part = ''  # w/ no dbname, Sqlite should use memory, which required 2 slash. Rel dir is 3, abs dir is 4

        dsn = '{}://{}{}'.format(d['driver'], host_part, path_part)

    return config, dsn