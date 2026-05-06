def get_version():
    """
    Get project version (using versioneer)
    :return: string containing version
    """
    setup_versioneer()
    clean_cache()
    import versioneer
    version = versioneer.get_version()
    parsed_version = parse_version(version)
    if '*@' in str(parsed_version):
        import time
        version += str(int(time.time()))
    return version