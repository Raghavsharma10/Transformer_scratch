def error(message, code='ERROR'):
    """Display Error.

    Method prints the error message, message being given
    as an input.

    Arguments:
        message {string} -- The message to be displayed.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output = now + ' [' + torn.plugins.colors.FAIL + \
            code + torn.plugins.colors.ENDC + '] \t' + \
            message
    print(output)