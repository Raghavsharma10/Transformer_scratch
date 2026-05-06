def roman_to_int(roman_string):
    """
    Converts a string of roman numbers into an integer.

    .. code: python

        reusables.roman_to_int("XXXVI")
        # 36


    :param roman_string: XVI or similar
    :return: parsed integer
    """
    roman_string = roman_string.upper().strip()
    if "IIII" in roman_string:
        raise ValueError("Malformed roman string")
    value = 0
    skip_one = False
    last_number = None
    for i, letter in enumerate(roman_string):
        if letter not in _roman_dict:
            raise ValueError("Malformed roman string")
        if skip_one:
            skip_one = False
            continue
        if i < (len(roman_string) - 1):
            double_check = letter + roman_string[i + 1]
            if double_check in _roman_dict:
                if last_number and _roman_dict[double_check] > last_number:
                    raise ValueError("Malformed roman string")
                last_number = _roman_dict[double_check]
                value += _roman_dict[double_check]
                skip_one = True
                continue
        if last_number and _roman_dict[letter] > last_number:
            raise ValueError("Malformed roman string")
        last_number = _roman_dict[letter]
        value += _roman_dict[letter]
    return value