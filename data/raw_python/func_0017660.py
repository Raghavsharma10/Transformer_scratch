def main():
    """
    This is a Toil pipeline used to perform alignment of fastqs.
    """
    # Define Parser object and add to Toil
    if mock_mode():
        usage_msg = 'You have the TOIL_SCRIPTS_MOCK_MODE environment variable set, so this pipeline ' \
                    'will run in mock mode. To disable mock mode, set TOIL_SCRIPTS_MOCK_MODE=0'
    else:
        usage_msg = None

    parser = argparse.ArgumentParser(usage=usage_msg)
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('generate-config', help='Generates an editable config in the current working directory.')
    subparsers.add_parser('generate-manifest', help='Generates an editable manifest in the current working directory.')
    subparsers.add_parser('generate', help='Generates a config and manifest in the current working directory.')
    # Run subparser                                                                                                              
    parser_run = subparsers.add_parser('run', help='Runs the ADAM/GATK pipeline')
    default_config = 'adam-gatk-mock.config' if mock_mode() else 'adam-gatk.config'
    default_manifest = 'adam-gatk-mock-manifest.csv' if mock_mode() else 'adam-gatk-manifest.csv'
    parser_run.add_argument('--config', default=default_config, type=str,
                            help='Path to the (filled in) config file, generated with "generate-config".')
    parser_run.add_argument('--manifest', default=default_manifest,
                            type=str, help='Path to the (filled in) manifest file, generated with "generate-manifest". '
                                           '\nDefault value: "%(default)s".')
    Job.Runner.addToilOptions(parser_run)
    args = parser.parse_args()

    cwd = os.getcwd()
    if args.command == 'generate-config' or args.command == 'generate':
        generate_file(os.path.join(cwd, default_config), generate_config)
    if args.command == 'generate-manifest' or args.command == 'generate':
        generate_file(os.path.join(cwd, default_manifest), generate_manifest)
    # Pipeline execution
    elif args.command == 'run':
        require(os.path.exists(args.config), '{} not found. Please run '
                                             'generate-config'.format(args.config))
        if not hasattr(args, 'sample'):
            require(os.path.exists(args.manifest), '{} not found and no samples provided. Please '
                                                   'run "generate-manifest"'.format(args.manifest))
        # Parse config
        parsed_config = {x.replace('-', '_'): y for x, y in yaml.load(open(args.config).read()).iteritems()}
        inputs = argparse.Namespace(**parsed_config)

        # Parse manifest file
        uuid_list = []
        with open(args.manifest) as f_manifest:
            for line in f_manifest:
                if not line.isspace() and not line.startswith('#'):
                    uuid_list.append(line.strip())

        inputs.sort = False
        if not inputs.dir_suffix:
            inputs.dir_suffix = ''
        if not inputs.s3_bucket:
            inputs.s3_bucket = ''

        if inputs.master_ip and inputs.num_nodes:
            raise ValueError("Exactly one of master_ip (%s) and num_nodes (%d) must be provided." %
                             (inputs.master_ip, inputs.num_nodes))

        if not hasattr(inputs, 'master_ip') and inputs.num_nodes <= 1:
            raise ValueError('num_nodes allocates one Spark/HDFS master and n-1 workers, and thus must be greater '
                             'than 1. %d was passed.' % inputs.num_nodes)

        if (inputs.pipeline_to_run != "adam" and
            inputs.pipeline_to_run != "gatk" and
            inputs.pipeline_to_run != "both"):
            raise ValueError("pipeline_to_run must be either 'adam', 'gatk', or 'both'. %s was passed." % inputs.pipeline_to_run)

        Job.Runner.startToil(Job.wrapJobFn(sample_loop, uuid_list, inputs), args)