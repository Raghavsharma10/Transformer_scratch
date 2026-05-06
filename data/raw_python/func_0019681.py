def save(filepath=None, **kwargs):
    """
        Saves a list of keyword arguments as environment variables to a file.
        If no filepath given will default to the default `.env` file.
    """
    if filepath is None:
        filepath = os.path.join('.env')

    with open(filepath, 'wb') as file_handle:
        file_handle.writelines(
            '{0}={1}\n'.format(key.upper(), val)
            for key, val in kwargs.items()
        )