def add_arguments(parser):
  """Adds stock arguments to argparse parsers from scripts that submit grid
  jobs."""

  default_log_path = os.path.realpath('logs')

  parser.add_argument('--log-dir', metavar='LOG', type=str,
      dest='logdir', default=default_log_path,
      help='Base directory used for logging (defaults to "%(default)s")')

  q_choices = (
      'default', 'all.q',
      'q_1day', 'q1d',
      'q_1week', 'q1w',
      'q_1month', 'q1m',
      'q_1day_mth', 'q1dm',
      'q_1week_mth', 'q1wm',
      'q_gpu', 'gpu',
      'q_long_gpu', 'lgpu',
      'q_short_gpu', 'sgpu',
      )

  parser.add_argument('--queue-name', metavar='QUEUE', type=str,
      dest='queue', default=q_choices[0], choices=q_choices,
      help='Queue for submission - one of ' + \
          '|'.join(q_choices) + ' (defaults to "%(default)s")')

  parser.add_argument('--hostname', metavar='HOSTNAME', type=str,
      dest='hostname', default=None,
      help='If set, it asks the queue to use only a subset of the available nodes')
  parser.add_argument('--memfree', metavar='MEMFREE', type=str,
      dest='memfree', default=None,
      help='Adds the \'-l mem_free\' argument to qsub')
  parser.add_argument('--hvmem', metavar='HVMEM', type=str,
      dest='hvmem', default=None,
      help='Adds the \'-l h_vmem\' argument to qsub')
  parser.add_argument('--pe-opt', metavar='PE_OPT', type=str,
      dest='pe_opt', default=None,
      help='Adds the \'--pe \' argument to qsub')

  parser.add_argument('--no-cwd', default=True, action='store_false',
      dest='cwd', help='Do not change to the current directory when starting the grid job')

  parser.add_argument('--dry-run', default=False, action='store_true',
      dest='dryrun', help='Does not really submit anything, just print what would do instead')

  parser.add_argument('--job-database', default=None,
      dest='statefile', help='The path to the state file that will be created with the submissions (defaults to the parent directory of your logs directory)')

  return parser