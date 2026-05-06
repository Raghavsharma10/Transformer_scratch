def text2wngram(text, output_file, n=3, chars=63636363, words=9090909, compress=False, verbosity=2):
    """
        List of every word n-gram which occurred in the text, along with its number of occurrences.
        The maximum numbers of charactors and words that can be stored in the buffer are given by the chars and words parameters.
    """
    cmd = ['text2wngram']
    
    if n:
        cmd.extend(['-n', n])
    
    if chars:
        cmd.extend(['-chars', chars])

    if words:
        cmd.extend(['-words', words])

    if compress:
        cmd.append('-compress')
    
    if verbosity:
        cmd.extend(['-verbosity', verbosity])

    # Ensure that every parameter is of type 'str'
    cmd = [str(x) for x in cmd]

    with tempfile.SpooledTemporaryFile() as input_f:
        input_f.write(text.encode('utf-8') if sys.version_info >= (3,) and type(text) is str else text)
        input_f.seek(0)
        with open(output_file,'w+') as output_f:
            with  output_to_debuglogger() as err_f:
                with do_in_tempdir():
                    exitcode = subprocess.call(cmd, stdin=input_f, stdout=output_f, stderr=err_f)

    logger = logging.getLogger(__name__)
    logger.debug("Command '%s' returned with exit code '%d'." % (' '.join(cmd), exitcode))

    if exitcode != 0:
        raise ConversionError("'%s' returned with non-zero exit status '%s'" % (cmd[0], exitcode))