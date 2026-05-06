def ask_path(question: str, default: str = None) -> str:
    """Asks for a path"""
    default_q = " [default: {0}]: ".format(
        default) if default is not None else ""
    answer = input("{0} [{1}]: ".format(question, default_q))

    if answer == "":
        return default

    if os.path.isdir(answer):
        return answer

    print(
        "No such directory: {answer}, please try again".format(answer=answer))
    return ask_path(question, default)