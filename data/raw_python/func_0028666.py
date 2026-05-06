def get_target_hash(target_filepath):
  """
  <Purpose>
    Compute the hash of 'target_filepath'. This is useful in conjunction with
    the "path_hash_prefixes" attribute in a delegated targets role, which tells
    us which paths it is implicitly responsible for.

    The repository may optionally organize targets into hashed bins to ease
    target delegations and role metadata management.  The use of consistent
    hashing allows for a uniform distribution of targets into bins.

  <Arguments>
    target_filepath:
      The path to the target file on the repository. This will be relative to
      the 'targets' (or equivalent) directory on a given mirror.

  <Exceptions>
    None.

  <Side Effects>
    None.

  <Returns>
    The hash of 'target_filepath'.
  """

  # Does 'target_filepath' have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.RELPATH_SCHEMA.check_match(target_filepath)

  # Calculate the hash of the filepath to determine which bin to find the
  # target.  The client currently assumes the repository uses
  # 'HASH_FUNCTION' to generate hashes and 'utf-8'.
  digest_object = securesystemslib.hash.digest(HASH_FUNCTION)
  encoded_target_filepath = target_filepath.encode('utf-8')
  digest_object.update(encoded_target_filepath)
  target_filepath_hash = digest_object.hexdigest()

  return target_filepath_hash