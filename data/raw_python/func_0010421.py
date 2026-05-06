def find_executable(executable):
    '''
    Finds executable in PATH

    Returns:
        string or None
    '''
    logger = logging.getLogger(__name__)
    logger.debug("Checking executable '%s'...", executable)
    executable_path = _find_executable(executable)
    found = executable_path is not None
    if found:
        logger.debug("Executable '%s' found: '%s'", executable, executable_path)
    else:
        logger.debug("Executable '%s' not found", executable)
    return executable_path