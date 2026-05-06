def version():
    """Get the version number without importing the mrcfile package."""
    namespace = {}
    with open(os.path.join('mrcfile', 'version.py')) as f:
        exec(f.read(), namespace)
    return namespace['__version__']