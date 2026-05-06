def import_ed25519_privatekey_from_file(filepath, password=None, prompt=False):
  """
  <Purpose>
    Import the encrypted ed25519 key file in 'filepath', decrypt it, and return
    the key object in 'securesystemslib.formats.ED25519KEY_SCHEMA' format.

    The private key (may also contain the public part) is encrypted with AES
    256 and CTR the mode of operation.  The password is strengthened with
    PBKDF2-HMAC-SHA256.

  <Arguments>
    filepath:
      <filepath> file, an RSA encrypted key file.

    password:
      The password, or passphrase, to import the private key (i.e., the
      encrypted key file 'filepath' must be decrypted before the ed25519 key
      object can be returned.

    prompt:
      If True the user is prompted for a passphrase to decrypt 'filepath'.
      Default is False.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted or the imported key object contains an invalid key type (i.e.,
    not 'ed25519').

    securesystemslib.exceptions.CryptoError, if 'filepath' cannot be decrypted.

  <Side Effects>
    'password' is used to decrypt the 'filepath' key file.

  <Returns>
    An ed25519 key object of the form:
    'securesystemslib.formats.ED25519KEY_SCHEMA'.
  """

  # Does 'filepath' have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

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
        confirm=False)

    # If user sets an empty string for the password, explicitly set the
    # password to None, because some functions may expect this later.
    if len(password) == 0: # pragma: no cover
      password = None

  # Finally, regardless of password, try decrypting the key, if necessary.
  # Otherwise, load it straight from the disk.
  with open(filepath, 'rb') as file_object:
    json_str = file_object.read()
    return securesystemslib.keys.\
           import_ed25519key_from_private_json(json_str, password=password)