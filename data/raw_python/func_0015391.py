def check_sysdeps(vext_files):
    """
    Check that imports in 'test_imports' succeed
    otherwise display message in 'install_hints'
    """

    @run_in_syspy
    def run(*modules):
        result = {}
        for m in modules:
            if m:
                try:
                    __import__(m)
                    result[m] = True
                except ImportError:
                    result[m] = False
        return result

    success = True
    for vext_file in vext_files:
        with open(vext_file) as f:
            vext = open_spec(f)
            install_hint = " ".join(vext.get('install_hints', ['System dependencies not found']))

            modules = vext.get('test_import', '')
            logger.debug("%s test imports of: %s", vext_file, modules)
            result = run(*modules)
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                for k, v in result.items():
                    logger.debug("%s: %s", k, v)
            if not all(result.values()):
                success = False
                print(install_hint)
    return success