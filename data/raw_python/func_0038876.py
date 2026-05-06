def is_owner(package, abspath):
    """Determine whether `abspath` belongs to `package`."""
    try:
        files = package['files']
        location = package['location']
    except KeyError:
        return False

    paths = (os.path.abspath(os.path.join(location, f))
             for f in files)

    return abspath in paths