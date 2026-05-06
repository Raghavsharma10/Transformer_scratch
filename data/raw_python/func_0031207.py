def main(args=None):
    """Download all .sra from NCBI SRA for a given experiment ID.

    Parameters
    ----------
    args: argparse.Namespace object, optional
        The argument values. If not specified, the values will be obtained by
        parsing the command line arguments using the `argparse` module.

    Returns
    -------
    int
        Exit code (0 if no error occurred).
    """
    if args is None:
        # parse command-line arguments
        parser = get_argument_parser()
        args = parser.parse_args()

    experiment_file = args.experiment_file
    output_file = args.output_file

    # log_file = args.log_file
    # quiet = args.quiet
    # verbose = args.verbose

    # logger = misc.get_logger(log_file=log_file, quiet=quiet,
    #                          verbose=verbose)

    host = 'ftp-trace.ncbi.nlm.nih.gov'
    user = 'anonymous'
    password = 'anonymous'

    # output_dir = download_dir + experiment_id + '/'
    # make sure output directory exists
    # misc.make_sure_dir_exists(output_dir)
    # logger.info('Created output directory: "%s".', output_dir)

    experiments = misc.read_single(experiment_file)

    runs = []
    with ftputil.FTPHost(host, user, password) as ftp_host:
        for exp in experiments:
            exp_dir = '/sra/sra-instant/reads/ByExp/sra/SRX/%s/%s/' \
                    % (exp[:6], exp)
            ftp_host.chdir(exp_dir)
            run_folders = ftp_host.listdir(ftp_host.curdir)
            # logging.info('Found %d run folders.',len(run_folders))

            for folder in run_folders:
                files = ftp_host.listdir(folder)
                assert len(files) == 1
                runs.append((exp, folder))

    with open(output_file, 'wb') as ofh:
        writer = csv.writer(ofh, dialect='excel-tab',
                            lineterminator=os.linesep,
                            quoting=csv.QUOTE_NONE)
        for r in runs:
            writer.writerow(r)
        
    return 0