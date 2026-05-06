def wfreq2vocab(wfreq_file, output_file, top=None, gt=None, records=1000000, verbosity=2):
    """
        Takes a a word unigram file, as produced by text2wfreq and converts it to a vocabulary file.
        The top parameter allows the user to specify the size of the vocabulary; if the function is called with the parameter top=20000, then the vocabulary will consist of the most common 20,000 words.
        The gt parameter allows the user to specify the number of times that a word must occur to be included in the vocabulary; if the function is called with the parameter gt=10, then the vocabulary will consist of all the words which occurred more than 10 times.
        If neither the gt, nor the top parameters are specified, then the function runs with the default setting of taking the top 20,000 words.
        The records parameter (default: 1000000) allows the user to specify how many of the word and count records to allocate memory for. If the number of words in the input exceeds this number, then the function will fail and raise a ConversionError, but a high number will obviously result in a higher memory requirement.
    """
    cmd = ['wfreq2vocab', '-verbosity', verbosity,
                           '-records', records]

    # Ensure that every parameter is of type 'str'
    cmd = [str(x) for x in cmd]

    if top:
        cmd.extend(['-top',top])
    elif gt:
        cmd.extend(['-gt',gt])

    with open(wfreq_file,'r') as input_f:
        with open(output_file,'w+') as output_f:
            with  output_to_debuglogger() as err_f:
                exitcode = subprocess.call(cmd, stdin=input_f, stdout=output_f, stderr=err_f)

    logger = logging.getLogger(__name__)
    logger.debug("Command '%s' returned with exit code '%d'." % (' '.join(cmd), exitcode))

    if exitcode != 0:
        raise ConversionError("'%s' returned with non-zero exit status '%s'" % (cmd[0], exitcode))