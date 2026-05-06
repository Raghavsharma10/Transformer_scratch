def Main():
  """The main program function.

  Returns:
    bool: True if successful or False if not.
  """
  argument_parser = argparse.ArgumentParser(
      description='Validates dtFabric format definitions.')

  argument_parser.add_argument(
      'source', nargs='?', action='store', metavar='PATH', default=None,
      help=(
          'path of the file or directory containing the dtFabric format '
          'definitions.'))

  options = argument_parser.parse_args()

  if not options.source:
    print('Source value is missing.')
    print('')
    argument_parser.print_help()
    print('')
    return False

  if not os.path.exists(options.source):
    print('No such file: {0:s}'.format(options.source))
    print('')
    return False

  logging.basicConfig(
      level=logging.INFO, format='[%(levelname)s] %(message)s')


  source_is_directory = os.path.isdir(options.source)

  validator = DefinitionsValidator()

  if source_is_directory:
    source_description = os.path.join(options.source, '*.yaml')
  else:
    source_description = options.source

  print('Validating dtFabric definitions in: {0:s}'.format(source_description))
  if source_is_directory:
    result = validator.CheckDirectory(options.source)
  else:
    result = validator.CheckFile(options.source)

  if not result:
    print('FAILURE')
  else:
    print('SUCCESS')

  return result