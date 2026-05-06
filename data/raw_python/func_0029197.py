def check_gcdt_update():
    """Check whether a newer gcdt is available and output a warning.

    """
    try:
        inst_version, latest_version = get_package_versions('gcdt')
        if inst_version < latest_version:
            log.warn('Please consider an update to gcdt version: %s' %
                                 latest_version)
    except GracefulExit:
        raise
    except Exception:
        log.warn('PyPi appears to be down - we currently can\'t check for newer gcdt versions')