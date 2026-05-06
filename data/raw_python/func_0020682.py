def main():
  """
  Parse command line arguments and then run the test suite
  """
  parser = argparse.ArgumentParser(description='A distributed test framework')
  parser.add_argument('testfile',
      help='The file that is used to determine the test suite run')
  parser.add_argument('--test-only',
      nargs='*',
      dest='test_list',
      help='run only the named tests to help debug broken tests')
  parser.add_argument('--machine-list',
      nargs='*',
      dest='machine_list',
      help='''mapping of logical host names to physical names allowing the same
              test suite to run on different hardware, each argument is a pair
              of logical name and physical name separated by a =''')
  parser.add_argument('--config-overrides',
      nargs='*',
      dest='config_overrides',
      help='''config overrides at execution time, each argument is a config with
              its value separated by a =. This has the highest priority of all
              configs''')
  parser.add_argument('-d', '--output-dir',
      dest='output_dir',
      help='''Directory to write output files and logs. Defaults to the current
              directory.''')
  parser.add_argument("--log-level", dest="log_level", help="Log level (default INFO)", default="INFO")
  parser.add_argument("--console-log-level", dest="console_level", help="Console Log level (default ERROR)",
                      default="ERROR")
  parser.add_argument("--nopassword", action='store_true', dest="nopassword", help="Disable password prompt")
  parser.add_argument("--user", dest="user", help="user to run the test as (defaults to current user)")
  args = parser.parse_args()
  try:
    call_main(args)
  except ValueError:
    #We only sys.exit here, as call_main is used as part of a unit test
    #and should not exit the system
    sys.exit(1)