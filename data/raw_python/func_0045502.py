def info(message, code='INFO'):
    """Display Information.

    Method prints the information message, message being given
    as an input.

    Arguments:
        message {string} -- The message to be displayed.
    """

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output = now + ' [' + torn.plugins.colors.OKBLUE + \
            code + torn.plugins.colors.ENDC + '] \t' + \
            message
    print(output)