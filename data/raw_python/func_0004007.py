def command_u2a(string, vargs):
    """
    Print the ARPABEY ASCII string corresponding to the given Unicode IPA string. 

    :param str string: the string to act upon
    :param dict vargs: the command line arguments
    """
    try:
        l = ARPABETMapper().map_unicode_string(
            unicode_string=string,
            ignore=vargs["ignore"],
            single_char_parsing=vargs["single_char_parsing"],
            return_as_list=True
        )
        print(vargs["separator"].join(l))
    except ValueError as exc:
        print_error(str(exc))