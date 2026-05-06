def ask_float(msg="Enter a float", dft=None, vld=None, hlp=None):
    """Prompts the user for a float."""
    vld = vld or [float]
    return ask(msg, dft=dft, vld=vld, fmt=partial(cast, typ=float), hlp=hlp)