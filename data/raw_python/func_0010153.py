def print_text(args, infilenames, outfilename=None):
    """Print text content of infiles to stdout.

    Keyword arguments:
    args -- program arguments (dict)
    infilenames -- names of user-inputted and/or downloaded files (list)
    outfilename -- only used for interface purposes (None)
    """
    for infilename in infilenames:
        parsed_text = get_parsed_text(args, infilename)
        if parsed_text:
            for line in parsed_text:
                print(line)
            print('')