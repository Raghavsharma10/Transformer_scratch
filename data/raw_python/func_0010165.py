def confirm_input(user_input):
    """Check user input for yes, no, or an exit signal."""
    if isinstance(user_input, list):
        user_input = ''.join(user_input)

    try:
        u_inp = user_input.lower().strip()
    except AttributeError:
        u_inp = user_input

    # Check for exit signal
    if u_inp in ('q', 'quit', 'exit'):
        sys.exit()
    if u_inp in ('y', 'yes'):
        return True
    return False