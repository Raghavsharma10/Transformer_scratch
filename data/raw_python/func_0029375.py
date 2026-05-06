def ask_int(msg="Enter an integer", dft=None, vld=None, hlp=None):
    """Prompts the user for an integer."""
    vld = vld or [int]
    return ask(msg, dft=dft, vld=vld, fmt=partial(cast, typ=int), hlp=hlp)