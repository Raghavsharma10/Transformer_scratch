def get_extra_commands():
    """Use the configuration to discover additional CLI packages to load"""
    from ambry.run import find_config_file
    from ambry.dbexceptions import ConfigurationError
    from ambry.util import yaml

    try:
        plugins_dir = find_config_file('cli.yaml')
    except ConfigurationError:
        return []

    with open(plugins_dir) as f:
        cli_modules = yaml.load(f)

    return cli_modules