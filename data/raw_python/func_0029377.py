def ask_str(msg="Enter a string", dft=None, vld=None, shw=True, blk=True, hlp=None):
    """Prompts the user for a string."""
    vld = vld or [str]
    return ask(msg, dft=dft, vld=vld, shw=shw, blk=blk, hlp=hlp)