def choices_label(choices: tuple, value) -> str:
    """
    Iterates (value,label) list and returns label matching the choice
    :param choices: [(choice1, label1), (choice2, label2), ...]
    :param value: Value to find
    :return: label or None
    """
    for key, label in choices:
        if key == value:
            return label
    return ''