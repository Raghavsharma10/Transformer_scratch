def cli(config, verbose, key_directory, no_verify, output_file, config_file):
    """
    Template and share OpenSSH ssh_config(5) files. A preprocessor for
    OpenSSH configurations.
    """
    config.verbose = verbose
    config.key_directory = key_directory
    config.config_file = config_file
    config.output_file = output_file
    config.no_verify = no_verify