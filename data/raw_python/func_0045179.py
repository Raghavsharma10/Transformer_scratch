def get_output_filename(args):
    """Returns a filename as string without an extension."""
    # If filename and path provided, use these for output text file.
    if args.directory is not None and args.fileprefix is not None:
        path = args.directory
        filename = args.fileprefix
        output = os.path.join(path, filename)
    # Otherwise, set output to current path
    elif args.fileprefix is not None:
        output = args.fileprefix
    else:
        output = args.pid
    return output