def write_files(args, infilenames, outfilename):
    """Write scraped or local file(s) in desired format.

    Keyword arguments:
    args -- program arguments (dict)
    infilenames -- names of user-inputted and/or downloaded files (list)
    outfilename -- name of output file (str)

    Remove PART(#).html files after conversion unless otherwise specified.
    """
    write_actions = {'print': utils.print_text,
                     'pdf': utils.write_pdf_files,
                     'csv': utils.write_csv_files,
                     'text': utils.write_text_files}
    try:
        for action in iterkeys(write_actions):
            if args[action]:
                write_actions[action](args, infilenames, outfilename)
    finally:
        if args['urls'] and not args['html']:
            utils.remove_part_files()