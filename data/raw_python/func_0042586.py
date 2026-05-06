def requireExecutables(executables):
    """
    Check that all of the given executables are on the path.
    If at least one of them is not, exit the script and inform
    the user of the missing requirement(s).
    """
    missingExecutables = []
    for executable in executables:
        if getPathOfExecutable(executable) is None:
            missingExecutables.append(executable)
    if len(missingExecutables) > 0:
        log("In order to run this script, the following "
            "executables need to be on the path:")
        for missingExecutable in missingExecutables:
            log(missingExecutable)
        exit(1)