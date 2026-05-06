def print_header(msg, sep='='):
    " More strong message "

    LOGGER.info("\n%s\n%s" % (msg, ''.join(sep for _ in msg)))