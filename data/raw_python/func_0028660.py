def get_file_details(filepath, hash_algorithms=['sha256']):
  """
  <Purpose>
    To get file's length and hash information.  The hash is computed using the
    sha256 algorithm.  This function is used in the signerlib.py and updater.py
    modules.

  <Arguments>
    filepath:
      Absolute file path of a file.

    hash_algorithms:

  <Exceptions>
    securesystemslib.exceptions.FormatError: If hash of the file does not match
    HASHDICT_SCHEMA.

    securesystemslib.exceptions.Error: If 'filepath' does not exist.

  <Returns>
    A tuple (length, hashes) describing 'filepath'.
  """

  # Making sure that the format of 'filepath' is a path string.
  # 'securesystemslib.exceptions.FormatError' is raised on incorrect format.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)
  securesystemslib.formats.HASHALGORITHMS_SCHEMA.check_match(hash_algorithms)

  # The returned file hashes of 'filepath'.
  file_hashes = {}

  # Does the path exists?
  if not os.path.exists(filepath):
    raise securesystemslib.exceptions.Error('Path ' + repr(filepath) + ' doest'
        ' not exist.')

  filepath = os.path.abspath(filepath)

  # Obtaining length of the file.
  file_length = os.path.getsize(filepath)

  # Obtaining hash of the file.
  for algorithm in hash_algorithms:
    digest_object = securesystemslib.hash.digest_filename(filepath, algorithm)
    file_hashes.update({algorithm: digest_object.hexdigest()})

  # Performing a format check to ensure 'file_hash' corresponds HASHDICT_SCHEMA.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.HASHDICT_SCHEMA.check_match(file_hashes)

  return file_length, file_hashes