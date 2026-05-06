def find_delegated_role(roles, delegated_role):
  """
  <Purpose>
    Find the index, if any, of a role with a given name in a list of roles.

  <Arguments>
    roles:
      The list of roles, each of which must have a 'name' attribute.

    delegated_role:
      The name of the role to be found in the list of roles.

  <Exceptions>
    securesystemslib.exceptions.RepositoryError, if the list of roles has
    invalid data.

  <Side Effects>
    No known side effects.

  <Returns>
    The unique index, an interger, in the list of roles.  if 'delegated_role'
    does not exist, 'None' is returned.
  """

  # Do the arguments have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.  Raise
  # 'securesystemslib.exceptions.FormatError' if any are improperly formatted.
  securesystemslib.formats.ROLELIST_SCHEMA.check_match(roles)
  securesystemslib.formats.ROLENAME_SCHEMA.check_match(delegated_role)

  # The index of a role, if any, with the same name.
  role_index = None

  for index in six.moves.xrange(len(roles)):
    role = roles[index]
    name = role.get('name')

    # This role has no name.
    if name is None:
      no_name_message = 'Role with no name.'
      raise securesystemslib.exceptions.RepositoryError(no_name_message)

    # Does this role have the same name?
    else:
      # This role has the same name, and...
      if name == delegated_role:
        # ...it is the only known role with the same name.
        if role_index is None:
          role_index = index

        # ...there are at least two roles with the same name.
        else:
          duplicate_role_message = 'Duplicate role (' + str(delegated_role) + ').'
          raise securesystemslib.exceptions.RepositoryError(
              'Duplicate role (' + str(delegated_role) + ').')

      # This role has a different name.
      else:
        logger.debug('Skipping delegated role: ' + repr(delegated_role))

  return role_index