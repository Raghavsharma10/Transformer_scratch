def ask(question, default=None):
    """
    @question: str
    @default: Any value which can be converted to string.

    Asks a user for a input.
    If default parameter is passed it will be appended to the end of the message in square brackets.
    """
    question = str(question)

    if default:
        question += ' [' + str(default) + ']'

    question += ': '

    reply = raw_input(question)
    return reply if reply else default