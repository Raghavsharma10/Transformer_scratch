def get_version(package):
    """Get version without importing the lib"""
    with io.open(os.path.join(BASE_DIR, package, '__init__.py'), encoding='utf-8') as fh:
        return [
            l.split('=', 1)[1].strip().strip("'").strip('"')
            for l in fh.readlines()
            if '__version__' in l][0]