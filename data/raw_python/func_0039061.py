def configure(username=None, password=None, overwrite=None, config_file=None):
    """Configure IA Mine with your Archive.org credentials."""
    username = input('Email address: ') if not username else username
    password = getpass('Password: ') if not password else password
    _config_file = write_config_file(username, password, overwrite, config_file)
    print('\nConfig saved to: {}'.format(_config_file))