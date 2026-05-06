def sed(file_path, pattern, replace_str, g=0):
    """Python impl of the bash sed command

    This method emulates the functionality of a bash sed command.

    :param file_path: (str) Full path to the file to be edited
    :param pattern: (str) Search pattern to replace as a regex
    :param replace_str: (str) String to replace the pattern
    :param g: (int) Whether to globally replace (0) or replace 1
        instance (equivalent to the 'g' option in bash sed
    :return: None
    :raises CommandError
    """
    log = logging.getLogger(mod_logger + '.sed')

    # Type checks on the args
    if not isinstance(file_path, basestring):
        msg = 'file_path argument must be a string'
        log.error(msg)
        raise CommandError(msg)
    if not isinstance(pattern, basestring):
        msg = 'pattern argument must be a string'
        log.error(msg)
        raise CommandError(msg)
    if not isinstance(replace_str, basestring):
        msg = 'replace_str argument must be a string'
        log.error(msg)
        raise CommandError(msg)

    # Ensure the file_path file exists
    if not os.path.isfile(file_path):
        msg = 'File not found: {f}'.format(f=file_path)
        log.error(msg)
        raise CommandError(msg)

    # Search for a matching pattern and replace matching patterns
    log.info('Updating file: %s...', file_path)
    for line in fileinput.input(file_path, inplace=True):
        if re.search(pattern, line):
            log.info('Updating line: %s', line)
            new_line = re.sub(pattern, replace_str, line, count=g)
            log.info('Replacing with line: %s', new_line)
            sys.stdout.write(new_line)
        else:
            sys.stdout.write(line)