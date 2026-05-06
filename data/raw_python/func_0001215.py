def ensure_databases_alive(max_retries: int = 100,
                           retry_timeout: int = 5,
                           exit_on_failure: bool = True) -> bool:
    """
    Checks every database alias in ``settings.DATABASES`` until it becomes available. After ``max_retries``
    attempts to reach any backend are failed it returns ``False``. If ``exit_on_failure`` is set it shuts down with
    ``exit(1)``.

    For every database alias it tries to ``SELECT 1``. If no errors raised it checks the next alias.

    :param exit_on_failure: set to ``True`` if there's no sense to continue
    :param int max_retries: number of attempts to reach every database; default is ``100``
    :param int retry_timeout: timeout in seconds between attempts
    :return: ``True`` if all backends are available, ``False`` if any backend check failed
    """
    template = """
    =============================
    Checking database connection `{CONNECTION}`:
        Engine: {ENGINE}
        Host: {HOST}
        Database: {NAME}
        User: {USER}
        Password: {PASSWORD}
    =============================\n"""
    for connection_name in connections:
        _db_settings = dict.fromkeys(['ENGINE', 'HOST', 'NAME', 'USER', 'PASSWORD'])
        _db_settings.update(settings.DATABASES[connection_name])
        _db_settings['CONNECTION'] = connection_name
        if _db_settings.get('PASSWORD'):
            _db_settings['PASSWORD'] = 'set'

        wf(template.format(**_db_settings))
        wf('Checking db connection alive... ', False)

        for i in range(max_retries):
            try:
                cursor = connections[connection_name].cursor()
                cursor.execute('SELECT 1')
                cursor.fetchone()

                wf('[+]\n')
                break
            except OperationalError as e:
                wf(str(e))
                sleep(retry_timeout)
        else:
            wf('Tried %s time(s). Shutting down.\n' % max_retries)
            exit_on_failure and exit(1)
            return False
    return True