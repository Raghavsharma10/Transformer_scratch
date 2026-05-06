def idngram2stats(input_file, output_file, n=3, fof_size=50, verbosity=2, ascii_input=False):
    """
        Lists the frequency-of-frequencies for each of the 2-grams, ... , n-grams, which can enable the user to choose appropriate cut-offs, and to specify appropriate memory requirements with the spec_num parameter in idngram2lm.
    """
    cmd = ['idngram2stats']
    if n:
        cmd.extend(['-n', n])
    
    if fof_size:
        cmd.extend(['-fof_size'], fof_size)

    if verbosity:
        cmd.extend(['-verbosity'], verbosity)

    if ascii_input:
        cmd.append(['-ascii_input'])

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