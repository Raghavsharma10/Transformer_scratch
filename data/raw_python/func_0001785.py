def ask_list(question: str, default: list = None) -> list:
    """Asks for a comma seperated list of strings"""
    default_q = " [default: {0}]: ".format(
        ",".join(default)) if default is not None else ""
    answer = input("{0} [{1}]: ".format(question, default_q))

    if answer == "":
        return default
    return [ans.strip() for ans in answer.split(",")]