def to_bool(answer, default):
    """
    Converts user answer to boolean
    """
    answer = str(answer).lower()
    default = str(default).lower()

    if answer and answer in "yes":
        return True

    return False