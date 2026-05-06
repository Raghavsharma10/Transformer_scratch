def paths_are_consistent_with_hash_prefixes(paths, path_hash_prefixes):
  """
  <Purpose>
    Determine whether a list of paths are consistent with their alleged path
    hash prefixes. By default, the SHA256 hash function is used.

  <Arguments>
    paths:
      A list of paths for which their hashes will be checked.

    path_hash_prefixes:
      The list of path hash prefixes with which to check the list of paths.

  <Exceptions>
    securesystemslib.exceptions.FormatError:
      If the arguments are improperly formatted.

  <Side Effects>
    No known side effects.

  <Returns>
    A Boolean indicating whether or not the paths are consistent with the
    hash prefix.
  """

  # Do the arguments have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.  Raise
  # 'securesystemslib.exceptions.FormatError' if any are improperly formatted.
  securesystemslib.formats.RELPATHS_SCHEMA.check_match(paths)
  securesystemslib.formats.PATH_HASH_PREFIXES_SCHEMA.check_match(path_hash_prefixes)

  # Assume that 'paths' and 'path_hash_prefixes' are inconsistent until
  # proven otherwise.
  consistent = False

  # The format checks above ensure the 'paths' and 'path_hash_prefix' lists
  # have lengths greater than zero.
  for path in paths:
    path_hash = get_target_hash(path)

    # Assume that every path is inconsistent until proven otherwise.
    consistent = False

    for path_hash_prefix in path_hash_prefixes:
      if path_hash.startswith(path_hash_prefix):
        consistent = True
        break

    # This path has no matching path_hash_prefix. Stop looking further.
    if not consistent:
      break

  return consistent