def create_parser():
    """Argument parser. """
    parser = argparse.ArgumentParser(
        prog='riboplot.py', description='Plot and output read counts for a single transcript')

    required = parser.add_argument_group('required arguments')
    required.add_argument('-b', '--ribo_file', help='Ribo-Seq alignment file in BAM format', required=True)
    required.add_argument('-f', '--transcriptome_fasta', help='FASTA format file of the transcriptome', required=True)
    required.add_argument('-t', '--transcript_name', help='Transcript name', metavar='TEXT', required=True)

    # plot function - optional arguments
    parser.add_argument('-n', '--rna_file', help='RNA-Seq alignment file (BAM)')
    parser.add_argument('-l', '--read_lengths', help='Read lengths to consider (default: %(default)s). '
                        'Multiple read lengths should be separated by commas. If multiple read lengths '
                        'are specified, corresponding read offsets should also be specified. If you do '
                        'not wish to apply an offset, please input 0 for the corresponding read length',
                        default='0', type=ribocore.lengths_offsets)
    parser.add_argument('-s', '--read_offsets', help='Read offsets (default: %(default)s). '
                        'Multiple read offsets should be separated by commas',
                        default='0', type=ribocore.lengths_offsets)
    parser.add_argument('-c', '--color_scheme', help='Color scheme to use (default: %(default)s)',
                        choices=['default', 'colorbrewer', 'rgb', 'greyorfs'], default='default')
    parser.add_argument('-m', '--html_file', help='Output file for results (HTML)', default='riboplot.html')
    parser.add_argument('-o', '--output_path', help='Files are saved in this directory', default='output')
    parser.add_argument('-d', '--debug', help='Flag. Produce debug output', action='store_true')

    return parser