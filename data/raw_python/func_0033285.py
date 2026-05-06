def parse_command_line_parameters():
    """ Parses command line arguments """
    usage = 'usage: %prog [options] fasta_filepath'
    version = 'Version: %prog 0.1'
    parser = OptionParser(usage=usage, version=version)

    # A binary 'verbose' flag
    parser.add_option('-p', '--is_protein', action='store_true',
                      dest='is_protein', default=False,
                      help='Pass if building db of protein sequences [default:'
                           ' False, nucleotide db]')

    parser.add_option('-o', '--output_dir', action='store', type='string',
                      dest='output_dir', default=None,
                      help='the output directory [default: directory '
                           'containing input fasta_filepath]')

    opts, args = parser.parse_args()
    num_args = 1
    if len(args) != num_args:
        parser.error('Must provide single filepath to build database from.')

    return opts, args