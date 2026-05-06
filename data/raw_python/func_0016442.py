def equate_initial(name1, name2):
    """
    Evaluates whether names match, or one name is the initial of the other
    """
    if len(name1) == 0 or len(name2) == 0:
        return False

    if len(name1) == 1 or len(name2) == 1:
        return name1[0] == name2[0]

    return name1 == name2