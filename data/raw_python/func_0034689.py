def text2wfreq(text, output_file, hashtablesize=1000000, verbosity=2):
    """
        List of every word which occurred in the text, along with its number of occurrences.
        Notes : Uses a hash-table to provide an efficient method of counting word occurrences. Output list is not sorted (due to "randomness" of the hash-table), but can be easily sorted into the user's desired order by the UNIX sort command. In any case, the output does not need to be sorted in order to serve as input for wfreq2vocab. Higher values for the hashtablesize parameter require more memory, but can reduce computation time.
    """
    cmd = ['text2wfreq', '-hash', hashtablesize,
                         '-verbosity', verbosity]

    # Ensure that every parameter is of type 'str'
    cmd = [str(x) for x in cmd]

    with tempfile.SpooledTemporaryFile() as input_f:
        input_f.write(text.encode('utf-8') if sys.version_info >= (3,) and type(text) is str else text)
        input_f.seek(0)
        with open(output_file,'w+') as output_f:
            with  output_to_debuglogger() as err_f:
                exitcode = subprocess.call(cmd, stdin=input_f, stdout=output_f, stderr=err_f)
    
    logger = logging.getLogger(__name__)
    logger.debug("Command '%s' returned with exit code '%d'." % (' '.join(cmd), exitcode))

    if exitcode != 0:
        raise ConversionError("'%s' returned with non-zero exit status '%s'" % (cmd[0], exitcode))