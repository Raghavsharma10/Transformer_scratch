def mergeidngram(output_file, input_files, n=3, ascii_input=False, ascii_output=False):
    """
        Takes a set of id n-gram files (in either binary (by default) or ASCII (if specified) format - note that they should all be in the same format, however) and outputs a merged id N-gram.

        Notes : This function can also be used to convert id n-gram files between ascii and binary formats.
    """
    cmd = ['mergeidngram']
    if n:
        cmd.extend(['-n', n])
    
    if ascii_input:
        cmd.append('-ascii_input')
    
    if ascii_output:
        cmd.append('-ascii_output')

    if len(input_file) > 1:
        raise MergeError("mergeidngram needs at least 1 input file")

    cmd.extend(input_files)

    # Ensure that every parameter is of type 'str'
    cmd = [str(x) for x in cmd]

    with open(output_file,'w+') as output_f:
        with  output_to_debuglogger() as err_f:
            exitcode = subprocess.call(cmd, stdout=output_f, stderr=err_f)
    
    logger = logging.getLogger(__name__)
    logger.debug("Command '%s' returned with exit code '%d'." % (' '.join(cmd), exitcode))

    if exitcode != 0:
        raise ConversionError("'%s' returned with non-zero exit status '%s'" % (cmd[0], exitcode))