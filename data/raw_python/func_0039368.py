def logfile(targetfile="ros.log"):
    """
    Set the file for Quilt to log to
    targetfile:
    Change the file to log to.
    """
    log = logging.getLogger(__name__)
    log.basicConfig(filename=str(targetfile))