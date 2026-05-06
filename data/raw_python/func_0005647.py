def parse_dsn(dsn, default_port=5432, protocol='http://'):
    """
    Разбирает строку подключения к БД и возвращает список из (host, port,
    username, password, dbname)

    :param dsn: Строка подключения. Например: username@localhost:5432/dname
    :type: str
    :param default_port: Порт по-умолчанию
    :type default_port: int
    :params protocol
    :type protocol str
    :return: [host, port, username, password, dbname]
    :rtype: list
    """
    parsed = urlparse(protocol + dsn)
    return [
        parsed.hostname or 'localhost',
        parsed.port or default_port,
        unquote(parsed.username)
        if parsed.username is not None else getuser(),
        unquote(parsed.password) if parsed.password is not None else None,
        parsed.path.lstrip('/'),
    ]