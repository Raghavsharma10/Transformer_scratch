def digest_filename(filename, algorithm=DEFAULT_HASH_ALGORITHM,
    hash_library=DEFAULT_HASH_LIBRARY, normalize_line_endings=False):
  """
  <Purpose>
    Generate a digest object, update its hash using a file object
    specified by filename, and then return it to the caller.

  <Arguments>
    filename:
      The filename belonging to the file object to be used.

    algorithm:
      The hash algorithm (e.g., 'md5', 'sha1', 'sha256').

    hash_library:
      The library providing the hash algorithms (e.g., 'hashlib').

    normalize_line_endings:
      Whether or not to normalize line endings for cross-platform support.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are
    improperly formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if the given
    'algorithm' is unsupported.

    securesystemslib.exceptions.UnsupportedLibraryError, if the given
    'hash_library' is unsupported.

  <Side Effects>
    None.

  <Returns>
    Digest object (e.g., hashlib.new(algorithm)).
  """
  # Are the arguments properly formatted?  If not, raise
  # 'securesystemslib.exceptions.FormatError'.
  securesystemslib.formats.RELPATH_SCHEMA.check_match(filename)
  securesystemslib.formats.NAME_SCHEMA.check_match(algorithm)
  securesystemslib.formats.NAME_SCHEMA.check_match(hash_library)

  digest_object = None

  # Open 'filename' in read+binary mode.
  with open(filename, 'rb') as file_object:
    # Create digest_object and update its hash data from file_object.
    # digest_fileobject() raises:
    # securesystemslib.exceptions.UnsupportedAlgorithmError
    # securesystemslib.exceptions.UnsupportedLibraryError
    digest_object = digest_fileobject(
        file_object, algorithm, hash_library, normalize_line_endings)

  return digest_object