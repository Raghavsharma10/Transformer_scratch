def import_rsa_privatekey_from_file(filepath, password=None,
    scheme='rsassa-pss-sha256', prompt=False):
  """
  <Purpose>
    Import the PEM file in 'filepath' containing the private key.

    If password is passed use passed password for decryption.
    If prompt is True use entered password for decryption.
    If no password is passed and either prompt is False or if the password
    entered at the prompt is an empty string, omit decryption, treating the
    key as if it is not encrypted.
    If password is passed and prompt is True, an error is raised. (See below.)

    The returned key is an object in the
    'securesystemslib.formats.RSAKEY_SCHEMA' format.

  <Arguments>
    filepath:
      <filepath> file, an RSA encrypted PEM file.  Unlike the public RSA PEM
      key file, 'filepath' does not have an extension.

    password:
      The passphrase to decrypt 'filepath'.

    scheme:
      The signature scheme used by the imported key.

    prompt:
      If True the user is prompted for a passphrase to decrypt 'filepath'.
      Default is False.

  <Exceptions>
    ValueError, if 'password' is passed and 'prompt' is True.

    ValueError, if 'password' is passed and it is an empty string.

    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.FormatError, if the entered password is
    improperly formatted.

    IOError, if 'filepath' can't be loaded.

    securesystemslib.exceptions.CryptoError, if a password is available
    and 'filepath' is not a valid key file encrypted using that password.

    securesystemslib.exceptions.CryptoError, if no password is available
    and 'filepath' is not a valid non-encrypted key file.

  <Side Effects>
    The contents of 'filepath' are read, optionally decrypted, and returned.

  <Returns>
    An RSA key object, conformant to 'securesystemslib.formats.RSAKEY_SCHEMA'.

  """

  # Does 'filepath' have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

  # Is 'scheme' properly formatted?
  securesystemslib.formats.RSA_SCHEME_SCHEMA.check_match(scheme)

  if password and prompt:
    raise ValueError("Passing 'password' and 'prompt' True is not allowed.")

  # If 'password' was passed check format and that it is not empty.
  if password is not None:
    securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

    # TODO: PASSWORD_SCHEMA should be securesystemslib.schema.AnyString(min=1)
    if not len(password):
      raise ValueError('Password must be 1 or more characters')

  elif prompt:
    # Password confirmation disabled here, which should ideally happen only
    # when creating encrypted key files (i.e., improve usability).
    # It is safe to specify the full path of 'filepath' in the prompt and not
    # worry about leaking sensitive information about the key's location.
    # However, care should be taken when including the full path in exceptions
    # and log files.
    # NOTE: A user who gets prompted for a password, can only signal that the
    # key is not encrypted by entering no password in the prompt, as opposed
    # to a programmer who can call the function with or without a 'password'.
    # Hence, we treat an empty password here, as if no 'password' was passed.
    password = get_password('Enter a password for an encrypted RSA'
        ' file \'' + Fore.RED + filepath + Fore.RESET + '\': ',
        confirm=False) or None

  if password is not None:
    # This check will not fail, because a mal-formatted passed password fails
    # above and an entered password will always be a string (see get_password)
    # However, we include it in case PASSWORD_SCHEMA or get_password changes.
    securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  else:
    logger.debug('No password was given. Attempting to import an'
        ' unencrypted file.')

  # Read the contents of 'filepath' that should be a PEM formatted private key.
  with open(filepath, 'rb') as file_object:
    pem_key = file_object.read().decode('utf-8')

  # Convert 'pem_key' to 'securesystemslib.formats.RSAKEY_SCHEMA' format.
  # Raise 'securesystemslib.exceptions.CryptoError' if 'pem_key' is invalid.
  # If 'password' is None decryption will be omitted.
  rsa_key = securesystemslib.keys.import_rsakey_from_private_pem(pem_key,
      scheme, password)

  return rsa_key