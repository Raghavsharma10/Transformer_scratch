def digest_fileobject(file_object, algorithm=DEFAULT_HASH_ALGORITHM,
    hash_library=DEFAULT_HASH_LIBRARY, normalize_line_endings=False):
  """
  <Purpose>
    Generate a digest object given a file object.  The new digest object
    is updated with the contents of 'file_object' prior to returning the
    object to the caller.

  <Arguments>
    file_object:
      File object whose contents will be used as the data
      to update the hash of a digest object to be returned.

    algorithm:
      The hash algorithm (e.g., 'md5', 'sha1', 'sha256').

    hash_library:
      The library providing the hash algorithms (e.g., 'hashlib').

    normalize_line_endings: (default False)
      Whether or not to normalize line endings for cross-platform support.
      Note that this results in ambiguous hashes (e.g. 'abc\n' and 'abc\r\n'
      will produce the same hash), so be careful to only apply this to text
      files (not binary), when that equivalence is desirable and cannot result
      in easily-maliciously-corrupted files producing the same hash as a valid
      file.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are
    improperly formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if an unsupported
    hashing algorithm was specified via 'algorithm'.

    securesystemslib.exceptions.UnsupportedLibraryError, if an unsupported
    crypto library was specified via 'hash_library'.

  <Side Effects>
    None.

  <Returns>
    Digest object (e.g., hashlib.new(algorithm)).
  """

  # Are the arguments properly formatted?  If not, raise
  # 'securesystemslib.exceptions.FormatError'.
  securesystemslib.formats.NAME_SCHEMA.check_match(algorithm)
  securesystemslib.formats.NAME_SCHEMA.check_match(hash_library)

  # Digest object returned whose hash will be updated using 'file_object'.
  # digest() raises:
  # securesystemslib.exceptions.UnsupportedAlgorithmError
  # securesystemslib.exceptions.UnsupportedLibraryError
  digest_object = digest(algorithm, hash_library)

  # Defensively seek to beginning, as there's no case where we don't
  # intend to start from the beginning of the file.
  file_object.seek(0)

  # Read the contents of the file object in at most 4096-byte chunks.
  # Update the hash with the data read from each chunk and return after
  # the entire file is processed.
  while True:
    data = file_object.read(DEFAULT_CHUNK_SIZE)
    if not data:
      break

    if normalize_line_endings:
      while data[-1:] == b'\r':
        c = file_object.read(1)
        if not c:
          break

        data += c

      data = (
          data
          # First Windows
          .replace(b'\r\n', b'\n')
          # Then Mac
          .replace(b'\r', b'\n')
      )

    if not isinstance(data, six.binary_type):
      digest_object.update(data.encode('utf-8'))

    else:
      digest_object.update(data)

  return digest_object