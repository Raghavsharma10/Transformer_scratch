def get_version():
    """Get single-source __version__."""
    pkg_dir = get_package_dir()
    with open(os.path.join(pkg_dir, 'nestcheck/_version.py')) as ver_file:
        string = ver_file.read()
    return string.strip().replace('__version__ = ', '').replace('\'', '')