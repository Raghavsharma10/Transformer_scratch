def inspect(config_file, profile):
    """Inspect existing configuration/profile."""

    try:
        section = load_profile_from_files(
            [config_file] if config_file else None, profile)

        click.echo("Configuration file: {}".format(config_file if config_file else "auto-detected"))
        click.echo("Profile: {}".format(profile if profile else "auto-detected"))
        click.echo("---")
        for key, val in section.items():
            click.echo("{} = {}".format(key, val))

    except (ValueError, ConfigFileReadError, ConfigFileParseError) as e:
        click.echo(e)