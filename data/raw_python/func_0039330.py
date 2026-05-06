def letternum(letter):
    """
    Get The Number Corresponding To A Letter
    """
    if not isinstance(letter, str):
        raise TypeError("Invalid letter provided.")
    if not len(letter) == 1:
        raise ValueError("Invalid letter length provided.")
    letter = letter.lower()
    alphaletters = string.ascii_lowercase
    for i in range(len(alphaletters)):
        if letter[0] == alphaletters[i]:
            return i + 1