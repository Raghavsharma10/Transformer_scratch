def get_config(config_file):
    """Get configuration from a file."""
    def load(fp):
        try:
            return yaml.safe_load(fp)
        except yaml.YAMLError as e:
            sys.stderr.write(text_type(e))
            sys.exit(1)  # TODO document exit codes

    if config_file == '-':
        return load(sys.stdin)
    if not os.path.exists(config_file):
        sys.stderr.write('ERROR: Must either run next to config.yaml or'
            ' specify a config file.\n' + __doc__)
        sys.exit(2)
    with open(config_file) as fp:
        return load(fp)