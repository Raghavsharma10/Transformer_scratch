def install_importer():
    """
    If in a virtualenv then load spec files to decide which
    modules can be imported from system site-packages and
    install path hook.
    """
    logging.debug('install_importer')
    if not in_venv():
        logging.debug('No virtualenv active py:[%s]', sys.executable)
        return False

    if disable_vext:
        logging.debug('Vext disabled by environment variable')
        return False

    if GatekeeperFinder.PATH_TRIGGER not in sys.path:
        try:
            load_specs()
            sys.path.append(GatekeeperFinder.PATH_TRIGGER)
            sys.path_hooks.append(GatekeeperFinder)
        except Exception as e:
            """
            Dont kill other programmes because of a vext error
            """
            logger.info(str(e))
            if logger.getEffectiveLevel() == logging.DEBUG:
                raise

        logging.debug("importer installed")
        return True