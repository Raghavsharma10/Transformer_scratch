def repoExitError(message):
    """
    Exits the repo manager with error status.
    """
    wrapper = textwrap.TextWrapper(
        break_on_hyphens=False, break_long_words=False)
    formatted = wrapper.fill("{}: error: {}".format(sys.argv[0], message))
    sys.exit(formatted)