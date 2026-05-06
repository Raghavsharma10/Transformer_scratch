def encrypt_key(key_object, password):
  """
  <Purpose>
    Return a string containing 'key_object' in encrypted form. Encrypted
    strings may be safely saved to a file.  The corresponding decrypt_key()
    function can be applied to the encrypted string to restore the original key
    object.  'key_object' is a key (e.g., RSAKEY_SCHEMA, ED25519KEY_SCHEMA).
    This function relies on the pyca_crypto_keys.py module to perform the
    actual encryption.

    Encrypted keys use AES-256-CTR-Mode, and passwords are strengthened with
    PBKDF2-HMAC-SHA256 (100K iterations by default, but may be overriden in
    'securesystemslib.settings.PBKDF2_ITERATIONS' by the user).

    http://en.wikipedia.org/wiki/Advanced_Encryption_Standard
    http://en.wikipedia.org/wiki/CTR_mode#Counter_.28CTR.29
    https://en.wikipedia.org/wiki/PBKDF2

    >>> ed25519_key = generate_ed25519_key()
    >>> password = 'secret'
    >>> encrypted_key = encrypt_key(ed25519_key, password).encode('utf-8')
    >>> securesystemslib.formats.ENCRYPTEDKEY_SCHEMA.matches(encrypted_key)
    True

  <Arguments>
    key_object:
      A key (containing also the private key portion) of the form
      'securesystemslib.formats.ANYKEY_SCHEMA'

    password:
      The password, or passphrase, to encrypt the private part of the RSA
      key.  'password' is not used directly as the encryption key, a stronger
      encryption key is derived from it.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.CryptoError, if 'key_object' cannot be
    encrypted.

  <Side Effects>
    None.

  <Returns>
    An encrypted string of the form:
    'securesystemslib.formats.ENCRYPTEDKEY_SCHEMA'.
  """

  # Does 'key_object' have the correct format?
  # This check will ensure 'key_object' has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.ANYKEY_SCHEMA.check_match(key_object)

  # Does 'password' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  # Encrypted string of 'key_object'.  The encrypted string may be safely saved
  # to a file and stored offline.
  encrypted_key = None

  # Generate an encrypted string of 'key_object' using AES-256-CTR-Mode, where
  # 'password' is strengthened with PBKDF2-HMAC-SHA256.
  encrypted_key = securesystemslib.pyca_crypto_keys.encrypt_key(key_object, password)

  return encrypted_key