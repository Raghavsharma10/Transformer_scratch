def ensure_parent_dir(filename):
  """
  <Purpose>
    To ensure existence of the parent directory of 'filename'.  If the parent
    directory of 'name' does not exist, create it.

    Example: If 'filename' is '/a/b/c/d.txt', and only the directory '/a/b/'
    exists, then directory '/a/b/c/d/' will be created.

  <Arguments>
    filename:
      A path string.

  <Exceptions>
    securesystemslib.exceptions.FormatError: If 'filename' is improperly
    formatted.

  <Side Effects>
    A directory is created whenever the parent directory of 'filename' does not
    exist.

  <Return>
    None.
  """

  # Ensure 'filename' corresponds to 'PATH_SCHEMA'.
  # Raise 'securesystemslib.exceptions.FormatError' on a mismatch.
  securesystemslib.formats.PATH_SCHEMA.check_match(filename)

  # Split 'filename' into head and tail, check if head exists.
  directory = os.path.split(filename)[0]

  if directory and not os.path.exists(directory):
    # mode = 'rwx------'. 448 (decimal) is 700 in octal.
    os.makedirs(directory, 448)