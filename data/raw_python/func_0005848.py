def sysinit(systype, conf, project):
    """Outputs configuration for system initialization subsystem."""

    click.secho(get_config(
        systype,
        conf=ConfModule(conf).configurations[0],
        conf_path=conf,
        project_name=project,
    ))