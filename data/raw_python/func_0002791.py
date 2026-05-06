def _run(passedArgs=None, stderr=None, stdout=None, exitFn=None):
    """Executes the script, gets prefix/suffix from the command prompt and
    produces output on STDOUT. For help with command line options, invoke
    script with '--help'.
    """

    description = """Finds the next available file-name in a sequence.

    This program will create a file of zero size and will output the path to it
    on STDOUT. No files which exist will be altered in this operation and
    concurrent invocations of this program will return separate files. In case
    of conflict, this program will attempt to generate a new file name up to
    'maxattempts' number of times before failing. (See --max-attempts)

    The sequence will start from the base argument (See --base, default: 0).

    This program will look for the next file in the sequence ignoring any gaps.
    Hence, if the files "a.0.txt" and "a.3.txt" exist, then the next file
    returned will be "a.4.txt" when called with prefix="a." and suffix=".txt".

    Returns:
        Path of the file which follows the provided pattern and can be opened
        for writing.

        Otherwise, it prints an error (wrong path, drive full, illegal
        character in filename, etc.) to stderr and exits with a non-zero error
        code.
    """

    argParser = _argparse.ArgumentParser(
            description=description,
            formatter_class=_argparse.RawTextHelpFormatter)

    argParser.add_argument('prefix',
                           help='Prefix for the sequence of files.')
    argParser.add_argument('suffix',
                           help='Suffix for the sequence of files.',
                           nargs='?',
                           default='')
    argParser.add_argument('folder',
                           help='The folder where the file will be created.',
                           nargs='?',
                           default=_os.getcwd())
    argParser.add_argument('-m', '--max-attempts',
                           help='Number of attempts to make before giving up.',
                           default=10)
    argParser.add_argument('-b', '--base',
                           help='From where to start counting (default: 0).',
                           default=0)

    passedArgs = passedArgs if passedArgs is not None else _sys.argv[1:]
    args = argParser.parse_args(passedArgs)

    stdout = _sys.stdout if stdout is None else stdout
    stderr = _sys.stderr if stderr is None else stderr

    try:
        nextFile = findNextFile(args.folder, prefix=args.prefix,
                                suffix=args.suffix,
                                maxattempts=args.max_attempts,
                                base=args.base)
        # The newline will be converted to the correct platform specific line
        # ending as `sys.stdout` is opened in non-binary mode.
        # Hence, we do not need to explicitly print out `\r\n` for Windows.
        stdout.write(nextFile + u'\n')
    except OSError as e:
        stderr.write(_os.strerror(e.errno) + u'\n')
        _sys.exit(e.errno)