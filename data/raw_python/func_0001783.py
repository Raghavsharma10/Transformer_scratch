def ask_int(question: str, default: int = None) -> int:
    """Asks for a number in a question"""
    default_q = " [default: {0}]: ".format(
        default) if default is not None else ""
    answer = input("{0} [{1}]: ".format(question, default_q))

    if not answer:
        if default is None:
            print("No default set, try again.")
            return ask_int(question, default)
        return default

    if any(x not in "1234567890" for x in answer):
        print("Please enter only numbers (0-9).")
        return ask_int(question, default)

    return int(answer)