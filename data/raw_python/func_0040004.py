def ask(message, options):
    """Ask the message interactively, with the given possible responses"""
    while 1:
        if os.environ.get('PIP_NO_INPUT'):
            raise Exception('No input was expected ($PIP_NO_INPUT set); question: %s' % message)
        response = raw_input(message)
        response = response.strip().lower()
        if response not in options:
            print('Your response (%r) was not one of the expected responses: %s' % (
                response, ', '.join(options)))
        else:
            return response