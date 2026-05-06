def main():
    """
    GATK germline pipeline with variant filtering and annotation.
    """
    # Define Parser object and add to jobTree
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # Generate subparsers
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('generate-config',
                          help='Generates an editable config in the current working directory.')
    subparsers.add_parser('generate-manifest',
                          help='Generates an editable manifest in the current working directory.')
    subparsers.add_parser('generate',
                          help='Generates a config and manifest in the current working directory.')

    # Run subparser
    parser_run = subparsers.add_parser('run', help='Runs the GATK germline pipeline')
    parser_run.add_argument('--config',
                            required=True,
                            type=str,
                            help='Path to the (filled in) config file, generated with '
                                 '"generate-config".')
    parser_run.add_argument('--manifest',
                            type=str,
                            help='Path to the (filled in) manifest file, generated with '
                                 '"generate-manifest".\nDefault value: "%(default)s".')
    parser_run.add_argument('--sample',
                            default=None,
                            nargs=2,
                            type=str,
                            help='Input sample identifier and BAM file URL or local path')
    parser_run.add_argument('--output-dir',
                            default=None,
                            help='Path/URL to output directory')
    parser_run.add_argument('-s', '--suffix',
                            default=None,
                            help='Additional suffix to add to the names of the output files')
    parser_run.add_argument('--preprocess-only',
                            action='store_true',
                            help='Only runs preprocessing steps')

    Job.Runner.addToilOptions(parser_run)
    options = parser.parse_args()

    cwd = os.getcwd()
    if options.command == 'generate-config' or options.command == 'generate':
        generate_file(os.path.join(cwd, 'config-toil-germline.yaml'), generate_config)
    if options.command == 'generate-manifest' or options.command == 'generate':
        generate_file(os.path.join(cwd, 'manifest-toil-germline.tsv'), generate_manifest)
    elif options.command == 'run':
        # Program checks
        for program in ['curl', 'docker']:
            require(next(which(program)),
                    program + ' must be installed on every node.'.format(program))

        require(os.path.exists(options.config), '{} not found. Please run "generate-config"'.format(options.config))

        # Read sample manifest
        samples = []
        if options.manifest:
            samples.extend(parse_manifest(options.manifest))

        # Add BAM sample from command line
        if options.sample:
            uuid, url = options.sample
            # samples tuple: (uuid, url, paired_url, rg_line)
            # BAM samples should not have as paired URL or read group line
            samples.append(GermlineSample(uuid, url, None, None))

        require(len(samples) > 0,
                'No samples were detected in the manifest or on the command line')

        # Parse inputs
        inputs = {x.replace('-', '_'): y for x, y in
                  yaml.load(open(options.config).read()).iteritems()}

        required_fields = {'genome_fasta',
                           'output_dir',
                           'run_bwa',
                           'sorted',
                           'snp_filter_annotations',
                           'indel_filter_annotations',
                           'preprocess',
                           'preprocess_only',
                           'run_vqsr',
                           'joint_genotype',
                           'run_oncotator',
                           'cores',
                           'file_size',
                           'xmx',
                           'suffix'}

        input_fields = set(inputs.keys())
        require(input_fields > required_fields,
                'Missing config parameters:\n{}'.format(', '.join(required_fields - input_fields)))

        if inputs['output_dir'] is None:
            inputs['output_dir'] = options.output_dir

        require(inputs['output_dir'] is not None,
                'Missing output directory PATH/URL')

        if inputs['suffix'] is None:
            inputs['suffix'] = options.suffix if options.suffix else ''

        if inputs['preprocess_only'] is None:
            inputs['preprocess_only'] = options.preprocess_only

        if inputs['run_vqsr']:
            # Check that essential VQSR parameters are present
            vqsr_fields = {'g1k_snp', 'mills', 'dbsnp', 'hapmap', 'omni'}
            require(input_fields > vqsr_fields,
                    'Missing parameters for VQSR:\n{}'.format(', '.join(vqsr_fields - input_fields)))

        # Check that hard filtering parameters are present. If only running preprocessing steps, then we do
        # not need filtering information.
        elif not inputs['preprocess_only']:
            hard_filter_fields = {'snp_filter_name', 'snp_filter_expression',
                                  'indel_filter_name', 'indel_filter_expression'}
            require(input_fields > hard_filter_fields,
                    'Missing parameters for hard filtering:\n{}'.format(', '.join(hard_filter_fields - input_fields)))

            # Check for falsey hard filtering parameters
            for hard_filter_field in hard_filter_fields:
                require(inputs[hard_filter_field], 'Missing %s value for hard filtering, '
                                                   'got %s.' % (hard_filter_field, inputs[hard_filter_field]))

        # Set resource parameters
        inputs['xmx'] = human2bytes(inputs['xmx'])
        inputs['file_size'] = human2bytes(inputs['file_size'])
        inputs['cores'] = int(inputs['cores'])

        inputs['annotations'] = set(inputs['snp_filter_annotations'] + inputs['indel_filter_annotations'])

        # HaplotypeCaller test data for testing
        inputs['hc_output'] = inputs.get('hc_output', None)

        # It is a toil-scripts convention to store input parameters in a Namespace object
        config = argparse.Namespace(**inputs)

        root = Job.wrapJobFn(run_gatk_germline_pipeline, samples, config)
        Job.Runner.startToil(root, options)