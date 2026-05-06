def parse_command_line_parameters(argv=None):
    """ Parses command line arguments """
    usage =\
        'usage: %prog [options] input_sequences_filepath'
    version = 'Version: %prog ' + __version__
    parser = OptionParser(usage=usage, version=version)

    parser.add_option('-o', '--output_fp', action='store',
                      type='string', dest='output_fp', help='Path to store ' +
                      'output file [default: generated from input_sequences_filepath]')

    parser.add_option('-c', '--min_confidence', action='store',
                      type='float', dest='min_confidence', help='minimum confidence ' +
                      'level to return a classification [default: %default]')

    parser.set_defaults(verbose=False, min_confidence=0.80)

    opts, args = parser.parse_args(argv)
    if len(args) != 1:
        parser.error('Exactly one argument is required.')

    return opts, args