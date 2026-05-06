def execute_and_report(command, *args, **kwargs):
    """Execute a command with arguments and wait for output.

    If execution was successful function will return True,
    if not, it will log the output using standard logging and return False.
    """
    logging.info("Execute: %s %s" % (command, " ".join(args)))
    try:
        status, out, err = execute(command, *args, **kwargs)
        if status == 0:
            logging.info(
                "%s Finished successfully. Exit Code: 0.",
                os.path.basename(command),
            )
            return True
        else:
            try:
                logging.error(
                    "%s failed! Exit Code: %s\nOut: %s\nError: %s",
                    os.path.basename(command),
                    status,
                    out,
                    err,
                )
            except Exception as e:
                # This fails when some non ASCII characters are returned
                # from the application
                logging.error(
                    "%s failed [%s]! Exit Code: %s\nOut: %s\nError: %s",
                    e,
                    os.path.basename(command),
                    status,
                    repr(out),
                    repr(err),
                )
            return False
    except Exception:
        logging.exception(
            "%s failed! Exception thrown!", os.path.basename(command)
        )
        return False