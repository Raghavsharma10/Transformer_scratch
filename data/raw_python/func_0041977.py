def resource_path(package_name: str, relative_path: typing.Union[str, Path]) -> Path:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    relative_path = Path(relative_path)
    methods = [
        _get_from_dev,
        _get_from_package,
        _get_from_sys,
    ]
    for method in methods:
        path = method(package_name, relative_path)
        if path.exists():
            return path

    raise FileNotFoundError(relative_path)