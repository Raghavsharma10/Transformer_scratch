def _manage_cmd(cmd, settings=None):
    # type: () -> None
    """ Run django ./manage.py command manually.

    This function eliminates the need for having ``manage.py`` (reduces file
    clutter).
    """
    import sys
    from os import environ
    from peltak.core import conf
    from peltak.core import context
    from peltak.core import log

    sys.path.insert(0, conf.get('src_dir'))

    settings = settings or conf.get('django.settings', None)
    environ.setdefault("DJANGO_SETTINGS_MODULE", settings)

    args = sys.argv[0:-1] + cmd

    if context.get('pretend', False):
        log.info("Would run the following manage command:\n<90>{}", args)
    else:
        from django.core.management import execute_from_command_line
        execute_from_command_line(args)