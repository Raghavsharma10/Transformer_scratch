def ensure_all_targets_allowed(rolename, list_of_targets, parent_delegations):
  """
  <Purpose>
    Ensure that the list of targets specified by 'rolename' are allowed; this
    is determined by inspecting the 'delegations' field of the parent role of
    'rolename'.  If a target specified by 'rolename' is not found in the
    delegations field of 'metadata_object_of_parent', raise an exception.  The
    top-level role 'targets' is allowed to list any target file, so this
    function does not raise an exception if 'rolename' is 'targets'.

    Targets allowed are either exlicitly listed under the 'paths' field, or
    implicitly exist under a subdirectory of a parent directory listed under
    'paths'.  A parent role may delegate trust to all files under a particular
    directory, including files in subdirectories, by simply listing the
    directory (e.g., '/packages/source/Django/', the equivalent of
    '/packages/source/Django/*').  Targets listed in hashed bins are also
    validated (i.e., its calculated path hash prefix must be delegated by the
    parent role).

    TODO: Should the TUF spec restrict the repository to one particular
    algorithm when calcutating path hash prefixes (currently restricted to
    SHA256)?  Should we allow the repository to specify in the role dictionary
    the algorithm used for these generated hashed paths?

  <Arguments>
    rolename:
      The name of the role whose targets must be verified. This is a
      role name and should not end in '.json'.  Examples: 'root', 'targets',
      'targets/linux/x86'.

    list_of_targets:
      The targets of 'rolename', as listed in targets field of the 'rolename'
      metadata.  'list_of_targets' are target paths relative to the targets
      directory of the repository.  The delegations of the parent role are
      checked to verify that the targets of 'list_of_targets' are valid.

    parent_delegations:
      The parent delegations of 'rolename'.  The metadata object stores
      the allowed paths and path hash prefixes of child delegations in its
      'delegations' attribute.

  <Exceptions>
    securesystemslib.exceptions.FormatError:
      If any of the arguments are improperly formatted.

    securesystemslib.exceptions.ForbiddenTargetError:
      If the targets of 'metadata_role' are not allowed according to
      the parent's metadata file.  The 'paths' and 'path_hash_prefixes'
      attributes are verified.

    securesystemslib.exceptions.RepositoryError:
      If the parent of 'rolename' has not made a delegation to 'rolename'.

  <Side Effects>
    None.

  <Returns>
    None.
  """

  # Do the arguments have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.  Raise
  # 'securesystemslib.exceptions.FormatError' if any are improperly formatted.
  securesystemslib.formats.ROLENAME_SCHEMA.check_match(rolename)
  securesystemslib.formats.RELPATHS_SCHEMA.check_match(list_of_targets)
  securesystemslib.formats.DELEGATIONS_SCHEMA.check_match(parent_delegations)

  # Return if 'rolename' is 'targets'.  'targets' is not a delegated role.  Any
  # target file listed in 'targets' is allowed.
  if rolename == 'targets':
    return

  # The allowed targets of delegated roles are stored in the parent's metadata
  # file.  Iterate 'list_of_targets' and confirm they are trusted, or their
  # root parent directory exists in the role delegated paths, or path hash
  # prefixes, of the parent role.  First, locate 'rolename' in the 'roles'
  # attribute of 'parent_delegations'.
  roles = parent_delegations['roles']
  role_index = find_delegated_role(roles, rolename)

  # Ensure the delegated role exists prior to extracting trusted paths from
  # the parent's 'paths', or trusted path hash prefixes from the parent's
  # 'path_hash_prefixes'.
  if role_index is not None:
    role = roles[role_index]
    allowed_child_paths = role.get('paths')
    allowed_child_path_hash_prefixes = role.get('path_hash_prefixes')
    actual_child_targets = list_of_targets

    if allowed_child_path_hash_prefixes is not None:
      consistent = paths_are_consistent_with_hash_prefixes

      # 'actual_child_tarets' (i.e., 'list_of_targets') should have lenth
      # greater than zero due to the format check above.
      if not consistent(actual_child_targets,
                        allowed_child_path_hash_prefixes):
        message =  repr(rolename) + ' specifies a target that does not' + \
          ' have a path hash prefix listed in its parent role.'
        raise securesystemslib.exceptions.ForbiddenTargetError(message)

    elif allowed_child_paths is not None:
      # Check that each delegated target is either explicitly listed or a
      # parent directory is found under role['paths'], otherwise raise an
      # exception.  If the parent role explicitly lists target file paths in
      # 'paths', this loop will run in O(n^2), the worst-case.  The repository
      # maintainer will likely delegate entire directories, and opt for
      # explicit file paths if the targets in a directory are delegated to
      # different roles/developers.
      for child_target in actual_child_targets:
        for allowed_child_path in allowed_child_paths:
          if fnmatch.fnmatch(child_target, allowed_child_path):
            break

        else:
          raise securesystemslib.exceptions.ForbiddenTargetError(
              'Role ' + repr(rolename) + ' specifies'
              ' target' + repr(child_target) + ',' + ' which is not an allowed'
              ' path according to the delegations set by its parent role.')

    else:
      # 'role' should have been validated when it was downloaded.
      # The 'paths' or 'path_hash_prefixes' attributes should not be missing,
      # so raise an error in case this clause is reached.
      raise securesystemslib.exceptions.FormatError(repr(role) + ' did not'
          ' contain one of the required fields ("paths" or'
          ' "path_hash_prefixes").')

  # Raise an exception if the parent has not delegated to the specified
  # 'rolename' child role.
  else:
    raise securesystemslib.exceptions.RepositoryError('The parent role has'
        ' not delegated to ' + repr(rolename) + '.')