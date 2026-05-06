def startswith_field(field, prefix):
    """
    RETURN True IF field PATH STRING STARTS WITH prefix PATH STRING
    """
    if prefix.startswith("."):
        return True
        # f_back = len(field) - len(field.strip("."))
        # p_back = len(prefix) - len(prefix.strip("."))
        # if f_back > p_back:
        #     return False
        # else:
        #     return True

    if field.startswith(prefix):
        if len(field) == len(prefix) or field[len(prefix)] == ".":
            return True
    return False