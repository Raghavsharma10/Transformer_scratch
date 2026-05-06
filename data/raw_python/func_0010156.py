def write_text_files(args, infilenames, outfilename):
    """Write text file(s) to disk.

    Keyword arguments:
    args -- program arguments (dict)
    infilenames -- names of user-inputted and/or downloaded files (list)
    outfilename -- name of output text file (str)
    """
    if not outfilename.endswith('.txt'):
        outfilename = outfilename + '.txt'
    outfilename = overwrite_file_check(args, outfilename)

    all_text = []  # Text must be aggregated if writing to a single output file
    for i, infilename in enumerate(infilenames):
        parsed_text = get_parsed_text(args, infilename)
        if parsed_text:
            if args['multiple']:
                if not args['quiet']:
                    print('Attempting to write to {0}.'.format(outfilename))
                write_file(parsed_text, outfilename)
            elif args['single']:
                all_text += parsed_text
                # Newline added between multiple files being aggregated
                if len(infilenames) > 1 and i < len(infilenames) - 1:
                    all_text.append('\n')

    # Write all text to a single output file
    if args['single'] and all_text:
        if not args['quiet']:
            print('Attempting to write {0} page(s) to {1}.'
                  .format(len(infilenames), outfilename))
        write_file(all_text, outfilename)