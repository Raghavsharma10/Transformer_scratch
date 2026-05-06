def import_ecdsa_privatekey_from_file(filepath, password=None):
  """
  <Purpose>
    Import the encrypted ECDSA key file in 'filepath', decrypt it, and return
    the key object in 'securesystemslib.formats.ECDSAKEY_SCHEMA' format.

    The 'cryptography' library is currently supported and performs the actual
    cryptographic routine.

  <Arguments>
    filepath:
      <filepath> file, an ECDSA encrypted key file.

    password:
      The password, or passphrase, to import the private key (i.e., the
      encrypted key file 'filepath' must be decrypted before the ECDSA key
      object can be returned.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted or the imported key object contains an invalid key type (i.e.,
    not 'ecdsa-sha2-nistp256').

    securesystemslib.exceptions.CryptoError, if 'filepath' cannot be decrypted.

  <Side Effects>
    'password' is used to decrypt the 'filepath' key file.

  <Returns>
    An ECDSA key object of the form: 'securesystemslib.formats.ECDSAKEY_SCHEMA'.
  """

  # Does 'filepath' have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

  # If the caller does not provide a password argument, prompt for one.
  # Password confirmation disabled here, which should ideally happen only
  # when creating encrypted key files (i.e., improve usability).
  if password is None: # pragma: no cover

    # It is safe to specify the full path of 'filepath' in the prompt and not
    # worry about leaking sensitive information about the key's location.
    # However, care should be taken when including the full path in exceptions
    # and log files.
    password = get_password('Enter a password for the encrypted ECDSA'
        ' key (' + Fore.RED + filepath + Fore.RESET + '): ',
        confirm=False)

  # Does 'password' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  # Store the encrypted contents of 'filepath' prior to calling the decryption
  # routine.
  encrypted_key = None

  with open(filepath, 'rb') as file_object:
    encrypted_key = file_object.read()

  # Decrypt the loaded key file, calling the 'cryptography' library to generate
  # the derived encryption key from 'password'.  Raise
  # 'securesystemslib.exceptions.CryptoError' if the decryption fails.
  key_object = securesystemslib.keys.decrypt_key(encrypted_key.decode('utf-8'),
      password)

  # Raise an exception if an unexpected key type is imported.
  if key_object['keytype'] != 'ecdsa-sha2-nistp256':
    message = 'Invalid key type loaded: ' + repr(key_object['keytype'])
    raise securesystemslib.exceptions.FormatError(message)

  # Add "keyid_hash_algorithms" so that equal ecdsa keys with different keyids
  # can be associated using supported keyid_hash_algorithms.
  key_object['keyid_hash_algorithms'] = \
      securesystemslib.settings.HASH_ALGORITHMS

  return key_object