def ngram2mgram(input_file, output_file, n, m, words=False, ascii_idngram=False):
    """
        Takes either a word n-gram file, or an id n-gram file and outputs a file of the same type where m < n.
    """
    cmd = ['ngram2mgram', '-n', n,
                          '-m', m]

    if words and ascii_idngram:
        raise ConversionError("Parameters 'words' and 'ascii_idngram' cannot both be True")

    if words:
        cmd.append('-words')
    elif ascii_idngram:
        cmd.append('-ascii')
    else:
        cmd.append('-binary')

    # Ensure that every parameter is of type 'str'
    cmd = [str(x) for x in cmd]

    with open(input_file,'r') as input_f:
        with open(output_file,'w+') as output_f:
            with  output_to_debuglogger() as err_f:
                exitcode = subprocess.call(cmd, stdin=input_f, stdout=output_f, stderr=err_f)
    
    logger = logging.getLogger(__name__)
    logger.debug("Command '%s' returned with exit code '%d'." % (' '.join(cmd), exitcode))

    if exitcode != 0:
        raise ConversionError("'%s' returned with non-zero exit status '%s'" % (cmd[0], exitcode))