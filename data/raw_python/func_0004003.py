def command_canonize(string, vargs):
    """
    Print the canonical representation of the given string. 

    It will replace non-canonical compound characters
    with their canonical synonym.

    :param str string: the string to act upon
    :param dict vargs: the command line arguments
    """
    try:
        ipa_string = IPAString(
            unicode_string=string,
            ignore=vargs["ignore"],
            single_char_parsing=vargs["single_char_parsing"]
        )
        print(vargs["separator"].join([(u"%s" % c) for c in ipa_string]))
    except ValueError as exc:
        print_error(str(exc))