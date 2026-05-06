def get_static_directory():
    """Retrieves the full path of the static directory

    @return: Full path of the static directory
    """
    directory = templates_dir = os.path.join(os.path.dirname(__file__), 'static')
    return directory