def main():
    """
    Computational Genomics Lab, Genomics Institute, UC Santa Cruz
    Toil exome pipeline

    Perform variant / indel analysis given a pair of tumor/normal BAM files.
    Samples are optionally preprocessed (indel realignment and base quality score recalibration)
    The output of this pipeline is a tarball containing results from MuTect, MuSe, and Pindel.

    General usage:
    1. Type "toil-exome generate" to create an editable manifest and config in the current working directory.
    2. Parameterize the pipeline by editing the config.
    3. Fill in the manifest with information pertaining to your samples.
    4. Type "toil-exome run [jobStore]" to execute the pipeline.

    Please read the README.md located in the source directory or at:
    https://github.com/BD2KGenomics/toil-scripts/tree/master/src/toil_scripts/exome_variant_pipeline

    Structure of variant pipeline (per sample)

           1 2 3 4          14 -------
           | | | |          |        |
        0 --------- 5 ----- 15 -------- 17
                    |       |        |
                   ---      16 -------
                   | |
                   6 7
                   | |
                   8 9
                   | |
                  10 11
                   | |
                  12 13

    0 = Start node
    1 = reference index
    2 = reference dict
    3 = normal bam index
    4 = tumor bam index
    5 = pre-processing node / DAG declaration
    6,7 = RealignerTargetCreator
    8,9 = IndelRealigner
    10,11 = BaseRecalibration
    12,13 = PrintReads
    14 = MuTect
    15 = Pindel
    16 = MuSe
    17 = Consolidate Output and move/upload results
    ==================================================
    Dependencies
    Curl:       apt-get install curl
    Docker:     wget -qO- https://get.docker.com/ | sh
    Toil:       pip install toil
    Boto:       pip install boto (OPTIONAL)
    """
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    # Generate subparsers
    subparsers.add_parser('generate-config', help='Generates an editable config in the current working directory.')
    subparsers.add_parser('generate-manifest', help='Generates an editable manifest in the current working directory.')
    subparsers.add_parser('generate', help='Generates a config and manifest in the current working directory.')
    # Run subparser
    parser_run = subparsers.add_parser('run', help='Runs the CGL exome pipeline')
    parser_run.add_argument('--config', default='config-toil-exome.yaml', type=str,
                            help='Path to the (filled in) config file, generated with "generate-config". '
                                 '\nDefault value: "%(default)s"')
    parser_run.add_argument('--manifest', default='manifest-toil-exome.tsv', type=str,
                            help='Path to the (filled in) manifest file, generated with "generate-manifest". '
                                 '\nDefault value: "%(default)s"')
    parser_run.add_argument('--normal', default=None, type=str,
                            help='URL for the normal BAM. URLs can take the form: http://, ftp://, file://, s3://, '
                                 'and gnos://. The UUID for the sample must be given with the "--uuid" flag.')
    parser_run.add_argument('--tumor', default=None, type=str,
                            help='URL for the tumor BAM. URLs can take the form: http://, ftp://, file://, s3://, '
                                 'and gnos://. The UUID for the sample must be given with the "--uuid" flag.')
    parser_run.add_argument('--uuid', default=None, type=str, help='Provide the UUID of a sample when using the'
                                                                   '"--tumor" and "--normal" option')
    # If no arguments provided, print full help menu
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    # Add Toil options
    Job.Runner.addToilOptions(parser_run)
    args = parser.parse_args()
    # Parse subparsers related to generation of config and manifest
    cwd = os.getcwd()
    if args.command == 'generate-config' or args.command == 'generate':
        generate_file(os.path.join(cwd, 'config-toil-exome.yaml'), generate_config)
    if args.command == 'generate-manifest' or args.command == 'generate':
        generate_file(os.path.join(cwd, 'manifest-toil-exome.tsv'), generate_manifest)
    # Pipeline execution
    elif args.command == 'run':
        require(os.path.exists(args.config), '{} not found. Please run '
                                             '"toil-rnaseq generate-config"'.format(args.config))
        if args.normal or args.tumor or args.uuid:
            require(args.normal and args.tumor and args.uuid, '"--tumor", "--normal" and "--uuid" must all be supplied')
            samples = [[args.uuid, args.normal, args.tumor]]
        else:
            samples = parse_manifest(args.manifest)
        # Parse config
        parsed_config = {x.replace('-', '_'): y for x, y in yaml.load(open(args.config).read()).iteritems()}
        config = argparse.Namespace(**parsed_config)
        config.maxCores = int(args.maxCores) if args.maxCores else sys.maxint
        # Exome pipeline sanity checks
        if config.preprocessing:
            require(config.reference and config.phase and config.mills and config.dbsnp,
                    'Missing inputs for preprocessing, check config file.')
        if config.run_mutect:
            require(config.reference and config.dbsnp and config.cosmic,
                    'Missing inputs for MuTect, check config file.')
        if config.run_pindel:
            require(config.reference, 'Missing input (reference) for Pindel.')
        if config.run_muse:
            require(config.reference and config.dbsnp,
                    'Missing inputs for MuSe, check config file.')
        require(config.output_dir, 'No output location specified: {}'.format(config.output_dir))
        # Program checks
        for program in ['curl', 'docker']:
            require(next(which(program), None), program + ' must be installed on every node.'.format(program))

        # Launch Pipeline
        Job.Runner.startToil(Job.wrapJobFn(download_shared_files, samples, config), args)