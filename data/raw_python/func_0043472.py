def call(cmd, shell=True, **kwargs):
    " Run shell command. "

    LOGGER.debug("Cmd: %s" % cmd)
    check_call(cmd, shell=shell, stdout=LOGFILE_HANDLER.stream, **kwargs)