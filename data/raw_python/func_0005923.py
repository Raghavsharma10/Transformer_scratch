def run_uwsgi(config_section, compile_only=False):
    """Runs uWSGI using the given section configuration.

    :param Section config_section:
    :param bool compile_only: Do not run, only compile and output configuration file for run.

    """
    config = config_section.as_configuration()

    if compile_only:
        config.print_ini()
        return

    config_path = config.tofile()
    os.execvp('uwsgi', ['uwsgi', '--ini=%s' % config_path])