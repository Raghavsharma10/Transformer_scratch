def command_check(string, vargs):
    """
    Check if the given string is IPA valid.

    If the given string is not IPA valid,
    print the invalid characters.

    :param str string: the string to act upon
    :param dict vargs: the command line arguments
    """
    is_valid = is_valid_ipa(string)
    print(is_valid)
    if not is_valid:
        valid_chars, invalid_chars = remove_invalid_ipa_characters(
            unicode_string=string,
            return_invalid=True
        )
        print_invalid_chars(invalid_chars, vargs)