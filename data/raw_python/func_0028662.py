def file_in_confined_directories(filepath, confined_directories):
  """
  <Purpose>
    Check if the directory containing 'filepath' is in the list/tuple of
    'confined_directories'.

  <Arguments>
    filepath:
      A string representing the path of a file.  The following example path
      strings are viewed as files and not directories: 'a/b/c', 'a/b/c.txt'.

    confined_directories:
      A list, or a tuple, of directory strings.

  <Exceptions>
   securesystemslib.exceptions.FormatError: On incorrect format of the input.

  <Return>
    Boolean.  True, if path is either the empty string
    or in 'confined_paths'; False, otherwise.
  """

  # Do the arguments have the correct format?
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.RELPATH_SCHEMA.check_match(filepath)
  securesystemslib.formats.RELPATHS_SCHEMA.check_match(confined_directories)

  for confined_directory in confined_directories:
    # The empty string (arbitrarily chosen) signifies the client is confined
    # to all directories and subdirectories.  No need to check 'filepath'.
    if confined_directory == '':
      return True

    # Normalized paths needed, to account for up-level references, etc.
    # TUF clients have the option of setting the list of directories in
    # 'confined_directories'.
    filepath = os.path.normpath(filepath)
    confined_directory = os.path.normpath(confined_directory)

    # A TUF client may restrict himself to specific directories on the
    # remote repository.  The list of paths in 'confined_path', not including
    # each path's subdirectories, are the only directories the client will
    # download targets from.
    if os.path.dirname(filepath) == confined_directory:
      return True

  return False