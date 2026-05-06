def digest(algorithm=DEFAULT_HASH_ALGORITHM, hash_library=DEFAULT_HASH_LIBRARY):
  """
  <Purpose>
    Provide the caller with the ability to create digest objects without having
    to worry about crypto library availability or which library to use.  The
    caller also has the option of specifying which hash algorithm and/or
    library to use.

    # Creation of a digest object using defaults or by specifying hash
    # algorithm and library.
    digest_object = securesystemslib.hash.digest()
    digest_object = securesystemslib.hash.digest('sha384')
    digest_object = securesystemslib.hash.digest('sha256', 'hashlib')

    # The expected interface for digest objects.
    digest_object.digest_size
    digest_object.hexdigest()
    digest_object.update('data')
    digest_object.digest()

    # Added hash routines by this module.
    digest_object = securesystemslib.hash.digest_fileobject(file_object)
    digest_object = securesystemslib.hash.digest_filename(filename)

  <Arguments>
    algorithm:
      The hash algorithm (e.g., 'md5', 'sha1', 'sha256').

    hash_library:
      The crypto library to use for the given hash algorithm (e.g., 'hashlib').

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are
    improperly formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if an unsupported
    hashing algorithm is specified, or digest could not be generated with given
    the algorithm.

    securesystemslib.exceptions.UnsupportedLibraryError, if an unsupported
    library was requested via 'hash_library'.

  <Side Effects>
    None.

  <Returns>
    Digest object (e.g., hashlib.new(algorithm)).
  """

  # Are the arguments properly formatted?  If not, raise
  # 'securesystemslib.exceptions.FormatError'.
  securesystemslib.formats.NAME_SCHEMA.check_match(algorithm)
  securesystemslib.formats.NAME_SCHEMA.check_match(hash_library)

  # Was a hashlib digest object requested and is it supported?
  # If so, return the digest object.
  if hash_library == 'hashlib' and hash_library in SUPPORTED_LIBRARIES:
    try:
      return hashlib.new(algorithm)

    except ValueError:
      raise securesystemslib.exceptions.UnsupportedAlgorithmError(algorithm)

  # Was a pyca_crypto digest object requested and is it supported?
  elif hash_library == 'pyca_crypto' and hash_library in SUPPORTED_LIBRARIES: #pragma: no cover
    # TODO: Add support for pyca/cryptography's hashing routines.
    pass

  # The requested hash library is not supported.
  else:
    raise securesystemslib.exceptions.UnsupportedLibraryError('Unsupported'
        ' library requested.  Supported hash'
        ' libraries: ' + repr(SUPPORTED_LIBRARIES))