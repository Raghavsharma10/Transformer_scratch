def get_password(prompt='Password: ', confirm=False):
  """
  <Purpose>
    Return the password entered by the user.  If 'confirm' is True, the user is
    asked to enter the previously entered password once again.  If they match,
    the password is returned to the caller.

  <Arguments>
    prompt:
      The text of the password prompt that is displayed to the user.

    confirm:
      Boolean indicating whether the user should be prompted for the password
      a second time.  The two entered password must match, otherwise the
      user is again prompted for a password.

  <Exceptions>
    None.

  <Side Effects>
    None.

  <Returns>
    The password entered by the user.
  """

  # Are the arguments the expected type?
  # If not, raise 'securesystemslib.exceptions.FormatError'.
  securesystemslib.formats.TEXT_SCHEMA.check_match(prompt)
  securesystemslib.formats.BOOLEAN_SCHEMA.check_match(confirm)

  while True:
    # getpass() prompts the user for a password without echoing
    # the user input.
    password = getpass.getpass(prompt, sys.stderr)

    if not confirm:
      return password
    password2 = getpass.getpass('Confirm: ', sys.stderr)

    if password == password2:
      return password

    else:
      print('Mismatch; try again.')