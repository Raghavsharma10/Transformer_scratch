def getsyssitepackages():
    """
    :return: list of site-packages from system python
    """
    global _syssitepackages
    if not _syssitepackages:
        if not in_venv():
            _syssitepackages = get_python_lib()
            return _syssitepackages

        @run_in_syspy
        def run(*args):
            import site
            return site.getsitepackages()

        output = run()
        _syssitepackages = output
        logger.debug("system site packages: %s", _syssitepackages)
    return _syssitepackages