def replace_filehandler(logname, new_file, level=None, frmt=None):
    """
    This utility function will remove a previous Logger FileHandler, if one
    exists, and add a new filehandler.

    Parameters:
      logname
          The name of the log to reconfigure, 'openaccess_epub' for example
      new_file
          The file location for the new FileHandler
      level
          Optional. Level of FileHandler logging, if not used then the new
          FileHandler will have the same level as the old. Pass in name strings,
          'INFO' for example
      frmt
          Optional string format of Formatter for the FileHandler, if not used
          then the new FileHandler will inherit the Formatter of the old, pass
          in format strings, '%(message)s' for example

    It is best practice to use the optional level and frmt arguments to account
    for the case where a previous FileHandler does not exist. In the case that
    they are not used and a previous FileHandler is not found, then the level
    will be set logging.DEBUG and the frmt will be set to
    openaccess_epub.utils.logs.STANDARD_FORMAT as a matter of safety.
    """
    #Call up the Logger to get reconfigured
    log = logging.getLogger(logname)

    #Set up defaults and whether explicit for level
    if level is not None:
        level = get_level(level)
        explicit_level = True
    else:
        level = logging.DEBUG
        explicit_level = False

    #Set up defaults and whether explicit for frmt
    if frmt is not None:
        frmt = logging.Formatter(frmt)
        explicit_frmt = True
    else:
        frmt = logging.Formatter(STANDARD_FORMAT)
        explicit_frmt = False

    #Look for a FileHandler to replace, set level and frmt if not explicit
    old_filehandler = None
    for handler in log.handlers:
        #I think this is an effective method of detecting FileHandler
        if type(handler) == logging.FileHandler:
            old_filehandler = handler
            if not explicit_level:
                level = handler.level
            if not explicit_frmt:
                frmt = handler.formatter
            break

    #Set up the new FileHandler
    new_filehandler = logging.FileHandler(new_file)
    new_filehandler.setLevel(level)
    new_filehandler.setFormatter(frmt)

    #Add the new FileHandler
    log.addHandler(new_filehandler)

    #Remove the old FileHandler if we found one
    if old_filehandler is not None:
        old_filehandler.close()
        log.removeHandler(old_filehandler)