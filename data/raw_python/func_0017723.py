def main():
    """
    This Toil pipeline aligns reads and performs alternative splicing analysis.

    Please read the README.md located in the same directory for run instructions.
    """
    # Define Parser object and add to toil
    url_prefix = 'https://s3-us-west-2.amazonaws.com/cgl-pipeline-inputs/'
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='Path to configuration file for samples, one per line. UUID,URL_to_bamfile. '
                             'The URL may be a standard "http://", a "file://<abs_path>", or "s3://<bucket>/<key>"')
    parser.add_argument('--gtf', help='URL to annotation GTF file',
                        default=url_prefix + 'rnaseq_cgl/gencode.v23.annotation.gtf')
    parser.add_argument('--gtf-pickle', help='Pickled GTF file',
                        default=url_prefix + 'spladder/gencode.v23.annotation.gtf.pickle')
    parser.add_argument('--gtf-m53', help='M53 preprocessing annotation table',
                        default=url_prefix + 'spladder/gencode.v23.annotation.gtf.m53')
    parser.add_argument('--positions', help='URL to SNP positions over genes file (TSV)',
                        default=url_prefix + 'spladder/positions_fixed.tsv')
    parser.add_argument('--genome', help='URL to Genome fasta',
                        default=url_prefix + 'rnaseq_cgl/hg38_no_alt.fa')
    parser.add_argument('--genome-index', help='Index file (fai) of genome',
                        default=url_prefix + 'spladder/hg38_no_alt.fa.fai')
    parser.add_argument('--ssec', default=None, help='Path to master key used for downloading encrypted files.')
    parser.add_argument('--output-s3-dir', default=None, help='S3 Directory of the form: s3://bucket/directory')
    parser.add_argument('--output-dir', default=None, help='full path where final results will be output')
    parser.add_argument('--sudo', action='store_true', default=False,
                        help='Set flag if sudo is required to run Docker.')
    parser.add_argument('--star-index', help='URL to download STAR Index built from HG38/gencodev23 annotation.',
                        default=url_prefix + 'rnaseq_cgl/starIndex_hg38_no_alt.tar.gz')
    parser.add_argument('--fwd-3pr-adapter', help="Sequence for the FWD 3' Read Adapter.", default='AGATCGGAAGAG')
    parser.add_argument('--rev-3pr-adapter', help="Sequence for the REV 3' Read Adapter.", default='AGATCGGAAGAG')
    Job.Runner.addToilOptions(parser)
    args = parser.parse_args()
    # Sanity Checks
    if args.config:
        assert os.path.isfile(args.config), 'Config not found at: {}'.format(args.config)
    if args.ssec:
        assert os.path.isfile(args.ssec), 'Encryption key not found at: {}'.format(args.config)
    if args.output_s3_dir:
        assert args.output_s3_dir.startswith('s3://'), 'Wrong format for output s3 directory'
    # Program checks
    for program in ['curl', 'docker']:
        assert which(program), 'Program "{}" must be installed on every node.'.format(program)

    Job.Runner.startToil(Job.wrapJobFn(parse_input_samples, args), args)