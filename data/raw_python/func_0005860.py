def get_config(systype, conf, conf_path, runner=None, project_name=None):
    """Returns init system configuration file contents.

    :param str|unicode systype: System type alias, e.g. systemd, upstart
    :param Section|Configuration conf: Configuration/Section object.
    :param str|unicode conf_path: File path to a configuration file or a command producing such a configuration.
    :param str|unicode runner: Runner command to execute conf_path. Defaults to ``uwsgiconf`` runner.
    :param str|unicode project_name: Project name to override.
    :rtype: str|unicode

    """
    runner = runner or ('%s run' % Finder.uwsgiconf())
    conf_path = abspath(conf_path)

    if isinstance(conf, Configuration):
        conf = conf.sections[0]  # todo Maybe something more intelligent.

    tpl = dedent(TEMPLATES.get(systype)(conf=conf))

    formatted = tpl.strip().format(
        project=project_name or conf.project_name or basename(dirname(conf_path)),
        command='%s %s' % (runner, conf_path),
    )

    return formatted