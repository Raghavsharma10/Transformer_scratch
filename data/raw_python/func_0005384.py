def devserver(port, admin_port, clear):
    # type: (int, int, bool) -> None
    """ Run devserver.

    Args:
        port (int):
            Port on which the app will be served.
        admin_port (int):
            Port on which the admin interface is served.
        clear (bool):
            If set to **True**, clear the datastore on startup.
    """
    admin_port = admin_port or (port + 1)

    args = [
        '--port={}'.format(port),
        '--admin_port={}'.format(admin_port)
    ]

    if clear:
        args += ['--clear_datastore=yes']

    with conf.within_proj_dir():
        shell.run('dev_appserver.py . {args}'.format(args=' '.join(args)))