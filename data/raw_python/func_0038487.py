def dump_commands(commands, directory=None, sub_dir=None):
    """
    Dump SQL commands to .sql files.

    :param commands: List of SQL commands
    :param directory: Directory to dump commands to
    :param sub_dir: Sub directory
    :return: Directory failed commands were dumped to
    """
    print('\t' + str(len(commands)), 'failed commands')

    # Create dump_dir directory
    if directory and os.path.isfile(directory):
        dump_dir = set_dump_directory(os.path.dirname(directory), sub_dir)
        return_dir = dump_dir
    elif directory:
        dump_dir = set_dump_directory(directory, sub_dir)
        return_dir = dump_dir
    else:
        dump_dir = TemporaryDirectory().name
        return_dir = TemporaryDirectory()

    # Create list of (path, content) tuples
    command_filepath = [(fail, os.path.join(dump_dir, str(count) + '.sql')) for count, fail in enumerate(commands)]

    # Dump failed commands to text file in the same directory as the commands
    # Utilize's multiprocessing module if it is available
    timer = Timer()
    if MULTIPROCESS:
        pool = Pool(cpu_count())
        pool.map(write_text_tup, command_filepath)
        pool.close()
        print('\tDumped ', len(command_filepath), 'commands\n\t\tTime      : {0}'.format(timer.end),
              '\n\t\tMethod    : (multiprocessing)\n\t\tDirectory : {0}'.format(dump_dir))
    else:
        for tup in command_filepath:
            write_text_tup(tup)
        print('\tDumped ', len(command_filepath), 'commands\n\t\tTime      : {0}'.format(timer.end),
              '\n\t\tMethod    : (sequential)\n\t\tDirectory : {0}'.format(dump_dir))

    # Return base directory of dumped commands
    return return_dir