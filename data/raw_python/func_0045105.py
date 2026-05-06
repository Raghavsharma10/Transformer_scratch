def parse_arguments():
    """
    Grab options and json files.
    """
    max_procs = MAX_PROCS
    dry_run = DRY_RUN
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s * %(levelname)s * %(message)s')

    parser = argparse.ArgumentParser(description="""nestrun - substitute values
            into a template and run commands in parallel.""")
    parser.add_argument('-j', '--processes', '--local', dest='local_procs',
            type=int, help="""Run a maximum of N processes in parallel locally
            (default: %(default)s)""", metavar='N', default=MAX_PROCS)
    parser.add_argument('--template', dest='template',
            metavar="'template text'", help="""Command-execution template, e.g.
            bash {infile}. By default, nestrun executes the templatefile.""")
    parser.add_argument('--stop-on-error', action='store_true',
            default=False, help="""Terminate remaining processes if any process
            returns non-zero exit status (default: %(default)s)""")
    parser.add_argument('--template-file', dest='template_file', metavar="FILE",
            help='Command-execution template file path.')
    parser.add_argument('--save-cmd-file', dest='savecmd_file',
            help="""Name of the file that will contain the command that was
            executed.""")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument('--log-file', dest='log_file', default='log.txt',
            help="""Name of the file that will contain output of the executed
            command.""")
    log_group.add_argument('--no-log', dest="log_file", action="store_const",
            default='log.txt', const=os.devnull, help="""Don't create a log
            file""")
    parser.add_argument('--dry-run', action='store_true', help="""Dry run mode,
            does not execute commands.""", default=False)
    parser.add_argument('--summary-file', type=argparse.FileType('w'),
            help="""Write a summary of the run to the specified file""")

    ctrl_group = parser.add_argument_group('Control files')
    ctrl_group.add_argument('json_files', metavar='control_files', type=extant_file,
            nargs='*', help="""Nestly control dictionaries""")
    ctrl_group.add_argument('-d', '--directory', help="""Run on all control
            files under %(metavar)s. May be used in place of specifying control
            files.""", metavar='DIR')
    arguments = parser.parse_args()


    # Load controls
    if bool(arguments.directory) == bool(arguments.json_files):
        parser.error('Exactly one of `-d` and control_files must be specified.')
    elif arguments.directory:
        arguments.json_files.extend(control_iter(arguments.directory))

    template = arguments.template

    # Make sure that either a template or a template file was given
    if arguments.template_file:
        # if given a template file, the default is to run the input
        if not arguments.template:
            template = os.path.join('.',
                    os.path.basename(arguments.template_file))

            # If using the default argument, the template must be executable:
            if (not os.access(arguments.template_file, os.X_OK) and not
                    arguments.dry_run):
                parser.error(
                        "{0} is not executable. Specify a template.".format(
                    arguments.template_file))

    if not (arguments.template or arguments.template_file):
        parser.exit("Error: Please specify either a template "
                "or a template file")

    logging.info('Template: %s', template)

    if arguments.local_procs is not None:
        max_procs = arguments.local_procs

    # Create a dictionary that will be shared amongst all forked processes.
    data = {}
    data['dry_run'] = arguments.dry_run
    data['start_directory'] = os.getcwd()
    data['template'] = template
    data['template_file'] = arguments.template_file
    data['savecmd_file'] = arguments.savecmd_file
    data['log_file'] = arguments.log_file
    data['stop_on_error'] = arguments.stop_on_error
    data['summary_file'] = arguments.summary_file

    return data, max_procs, arguments.json_files