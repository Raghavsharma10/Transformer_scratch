def command_clean(string, vargs):
    """
    Remove characters that are not IPA valid from the given string,
    and print the remaining string.

    :param str string: the string to act upon
    :param dict vargs: the command line arguments
    """
    valid_chars, invalid_chars = remove_invalid_ipa_characters(
        unicode_string=string,
        return_invalid=True,
        single_char_parsing=vargs["single_char_parsing"]
    )
    print(u"".join(valid_chars))
    print_invalid_chars(invalid_chars, vargs)