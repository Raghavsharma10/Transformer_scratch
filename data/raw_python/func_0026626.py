def is_muted(what):
    """
    Checks if a logged event is to be muted for debugging purposes.

    Also goes through the solo list - only items in there will be logged!

    :param what:
    :return:
    """

    state = False

    for item in solo:
        if item not in what:
            state = True
        else:
            state = False
            break

    for item in mute:
        if item in what:
            state = True
            break

    return state