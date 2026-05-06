def warning(message, code='WARNING'):
    """Display Warning.

    Method prints the warning message, message being given
    as an input.

    Arguments:
        message {string} -- The message to be displayed.
    """

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output = now + ' [' + torn.plugins.colors.WARNING + \
            code + torn.plugins.colors.ENDC + '] \t' + \
            message
    print(output)