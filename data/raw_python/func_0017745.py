def main():
    """
    Computational Genomics Lab, Genomics Institute, UC Santa Cruz
    Toil BWA pipeline

    Alignment of fastq reads via BWA-kit

    General usage:
    1. Type "toil-bwa generate" to create an editable manifest and config in the current working directory.
    2. Parameterize the pipeline by editing the config.
    3. Fill in the manifest with information pertaining to your samples.
    4. Type "toil-bwa run [jobStore]" to execute the pipeline.

    Please read the README.md located in the source directory or at:
    https://github.com/BD2KGenomics/toil-scripts/tree/master/src/toil_scripts/bwa_alignment

    Structure of the BWA pipeline (per sample)

        0 --> 1

    0 = Download sample
    1 = Run BWA-kit
    ===================================================================
    :Dependencies:
    cURL:       apt-get install curl
    Toil:       pip install toil
    Docker:     wget -qO- https://get.docker.com/ | sh

    Optional:
    S3AM:       pip install --s3am (requires ~/.boto config file)
    Boto:       pip install boto
    """
    # Define Parser object and add to Toil
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    # Generate subparsers
    subparsers.add_parser('generate-config', help='Generates an editable config in the current working directory.')
    subparsers.add_parser('generate-manifest', help='Generates an editable manifest in the current working directory.')
    subparsers.add_parser('generate', help='Generates a config and manifest in the current working directory.')
    # Run subparser
    parser_run = subparsers.add_parser('run', help='Runs the BWA alignment pipeline')
    group = parser_run.add_mutually_exclusive_group()
    parser_run.add_argument('--config', default='config-toil-bwa.yaml', type=str,
                            help='Path to the (filled in) config file, generated with "generate-config".')
    group.add_argument('--manifest', default='manifest-toil-bwa.tsv', type=str,
                       help='Path to the (filled in) manifest file, generated with "generate-manifest". '
                            '\nDefault value: "%(default)s".')
    group.add_argument('--sample', nargs='+', action=required_length(2, 3),
                       help='Space delimited sample UUID and fastq files in the format: uuid url1 [url2].')
    # Print docstring help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    Job.Runner.addToilOptions(parser_run)
    args = parser.parse_args()
    # Parse subparsers related to generation of config and manifest
    cwd = os.getcwd()
    if args.command == 'generate-config' or args.command == 'generate':
        generate_file(os.path.join(cwd, 'config-toil-bwa.yaml'), generate_config)
    if args.command == 'generate-manifest' or args.command == 'generate':
        generate_file(os.path.join(cwd, 'manifest-toil-bwa.tsv'), generate_manifest)
    # Pipeline execution
    elif args.command == 'run':
        require(os.path.exists(args.config), '{} not found. Please run generate-config'.format(args.config))
        if not args.sample:
            args.sample = None
            require(os.path.exists(args.manifest), '{} not found and no sample provided. '
                                                   'Please run "generate-manifest"'.format(args.manifest))
        # Parse config
        parsed_config = {x.replace('-', '_'): y for x, y in yaml.load(open(args.config).read()).iteritems()}
        config = argparse.Namespace(**parsed_config)
        config.maxCores = int(args.maxCores) if args.maxCores else sys.maxint
        samples = [args.sample[0], args.sample[1:]] if args.sample else parse_manifest(args.manifest)
        # Sanity checks
        require(config.ref, 'Missing URL for reference file: {}'.format(config.ref))
        require(config.output_dir, 'No output location specified: {}'.format(config.output_dir))
        # Launch Pipeline
        Job.Runner.startToil(Job.wrapJobFn(download_reference_files, config, samples), args)