def ask_str(question: str, default: str = None):
    """Asks for a simple string"""
    default_q = " [default: {0}]: ".format(
        default) if default is not None else ""
    answer = input("{0} [{1}]: ".format(question, default_q))

    if answer == "":
        return default
    return answer