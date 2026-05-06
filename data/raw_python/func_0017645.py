def main():
    '''
    Sets up command line parser for Toil/ADAM based k-mer counter, and launches
    k-mer counter with optional Spark cluster.
    '''

    parser = argparse.ArgumentParser()

    # add parser arguments
    parser.add_argument('--input_path',
                        help='The full path to the input SAM/BAM/ADAM/FASTQ file.')
    parser.add_argument('--output-path',
                        help='full path where final results will be output.')
    parser.add_argument('--kmer-length',
                        help='Length to use for k-mer counting. Defaults to 20.',
                        default=20,
                        type=int)
    parser.add_argument('--spark-conf',
                        help='Optional configuration to pass to Spark commands. Either this or --workers must be specified.',
                        default=None)
    parser.add_argument('--memory',
                        help='Optional memory configuration for Spark workers/driver. This must be specified if --workers is specified.',
                        default=None,
                        type=int)
    parser.add_argument('--cores',
                        help='Optional core configuration for Spark workers/driver. This must be specified if --workers is specified.',
                        default=None,
                        type=int)
    parser.add_argument('--workers',
                        help='Number of workers to spin up in Toil. Either this or --spark-conf must be specified. If this is specified, --memory and --cores must be specified.',
                        default=None,
                        type=int)
    parser.add_argument('--sudo',
                        help='Run docker containers with sudo. Defaults to False.',
                        default=False,
                        action='store_true')

    Job.Runner.addToilOptions(parser)
    args = parser.parse_args()
    Job.Runner.startToil(Job.wrapJobFn(kmer_dag,
                                       args.kmer_length,
                                       args.input_path,
                                       args.output_path,
                                       args.spark_conf,
                                       args.workers,
                                       args.cores,
                                       args.memory,
                                       args.sudo,
                                       checkpoint=True), args)