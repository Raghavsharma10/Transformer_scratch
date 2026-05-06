def _run(command, quiet=False, timeout=None):
    """Run a command, returns command output."""
    try:
        with _spawn(command, quiet, timeout) as child:
            command_output = child.read().strip().replace("\r\n", "\n")
    except pexpect.TIMEOUT:
        logger.info(f"command {command} timed out")
        raise Error()

    return command_output