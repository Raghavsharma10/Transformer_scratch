def get_output_directory(args):
    """
    Determination of the directory for output placement involves possibilities
    for explicit user instruction (absolute path or relative to execution) and
    implicit default configuration (absolute path or relative to input) from
    the system global configuration file. This function is responsible for
    reliably returning the appropriate output directory which will contain any
    log(s), ePub(s), and unzipped output of OpenAccess_EPUB.

    It utilizes the parsed args, passed as an object, and is self-sufficient in
    accessing the config file.

    All paths returned by this function are absolute.
    """
    #Import the global config file as a module
    import imp
    config_path = os.path.join(cache_location(), 'config.py')
    try:
        config = imp.load_source('config', config_path)
    except IOError:
        print('Could not find {0}, please run oae-quickstart'.format(config_path))
        sys.exit()
    #args.output is the explicit user instruction, None if unspecified
    if args.output:
        #args.output may be an absolute path
        if os.path.isabs(args.output):
            return args.output  # return as is
        #or args.output may be a relative path, relative to cwd
        else:
            return evaluate_relative_path(relative=args.output)
    #config.default_output for default behavior without explicit instruction
    else:
        #config.default_output may be an absolute_path
        if os.path.isabs(config.default_output):
            return config.default_output
        #or config.default_output may be a relative path, relative to input
        else:
            if args.input:  # The case of single input
                if 'http://www' in args.input:
                    #Fetched from internet by URL
                    raw_name = url_input(args.input, download=False)
                    abs_input_path = os.path.join(os.getcwd(), raw_name+'.xml')
                elif args.input[:4] == 'doi:':
                    #Fetched from internet by DOI
                    raw_name = doi_input(args.input, download=False)
                    abs_input_path = os.path.join(os.getcwd(), raw_name+'.xml')
                else:
                    #Local option, could be anywhere
                    abs_input_path = get_absolute_path(args.input)
                abs_input_parent = os.path.split(abs_input_path)[0]
                return evaluate_relative_path(abs_input_parent, config.default_output)
            elif args.batch:  # The case of Batch Mode
                #Batch should only work on a supplied directory
                abs_batch_path = get_absolute_path(args.batch)
                return abs_batch_path
            elif args.zip:
                #Zip is a local-only option, behaves just like local xml
                abs_input_path = get_absolute_path(args.zip)
                abs_input_parent = os.path.split(abs_input_path)[0]
                return evaluate_relative_path(abs_input_parent, config.default_output)
            elif args.collection:
                return os.getcwd()
            else:  # Un-handled or currently unsupported options
                print('The output location could not be determined...')
                sys.exit()