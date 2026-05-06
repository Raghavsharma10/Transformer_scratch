def ask_bool(question: str, default: bool = True) -> bool:
    """Asks a question yes no style"""
    default_q = "Y/n" if default else "y/N"
    answer = input("{0} [{1}]: ".format(question, default_q))
    lower = answer.lower()
    if not lower:
        return default
    return lower == "y"