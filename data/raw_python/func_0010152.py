def overwrite_file_check(args, filename):
    """If filename exists, overwrite or modify it to be unique."""
    if not args['overwrite'] and os.path.exists(filename):
        # Confirm overwriting of the file, or modify filename
        if args['no_overwrite']:
            overwrite = False
        else:
            try:
                overwrite = confirm_input(input('Overwrite {0}? (yes/no): '
                                                .format(filename)))
            except (KeyboardInterrupt, EOFError):
                sys.exit()
        if not overwrite:
            new_filename = modify_filename_id(filename)
            while os.path.exists(new_filename):
                new_filename = modify_filename_id(new_filename)
            return new_filename
    return filename