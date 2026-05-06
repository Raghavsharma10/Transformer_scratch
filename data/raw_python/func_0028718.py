def decrypt_key(encrypted_key, passphrase):
  """
  <Purpose>
    Return a string containing 'encrypted_key' in non-encrypted form.  The
    decrypt_key() function can be applied to the encrypted string to restore
    the original key object, a key (e.g., RSAKEY_SCHEMA, ED25519KEY_SCHEMA).
    This function calls pyca_crypto_keys.py to perform the actual decryption.

    Encrypted keys use AES-256-CTR-Mode and passwords are strengthened with
    PBKDF2-HMAC-SHA256 (100K iterations be default, but may be overriden in
    'settings.py' by the user).

    http://en.wikipedia.org/wiki/Advanced_Encryption_Standard
    http://en.wikipedia.org/wiki/CTR_mode#Counter_.28CTR.29
    https://en.wikipedia.org/wiki/PBKDF2

    >>> ed25519_key = generate_ed25519_key()
    >>> password = 'secret'
    >>> encrypted_key = encrypt_key(ed25519_key, password)
    >>> decrypted_key = decrypt_key(encrypted_key.encode('utf-8'), password)
    >>> securesystemslib.formats.ANYKEY_SCHEMA.matches(decrypted_key)
    True
    >>> decrypted_key == ed25519_key
    True

  <Arguments>
    encrypted_key:
      An encrypted key (additional data is also included, such as salt, number
      of password iterations used for the derived encryption key, etc) of the
      form 'securesystemslib.formats.ENCRYPTEDKEY_SCHEMA'.  'encrypted_key'
      should have been generated with encrypt_key().

    password:
      The password, or passphrase, to decrypt 'encrypted_key'.  'password' is
      not used directly as the encryption key, a stronger encryption key is
      derived from it.  The supported general-purpose module takes care of
      re-deriving the encryption key.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.CryptoError, if 'encrypted_key' cannot be
    decrypted.

  <Side Effects>
    None.

  <Returns>
    A key object of the form: 'securesystemslib.formats.ANYKEY_SCHEMA' (e.g.,
    RSAKEY_SCHEMA, ED25519KEY_SCHEMA).
  """

  # Does 'encrypted_key' have the correct format?
  # This check ensures 'encrypted_key' has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.ENCRYPTEDKEY_SCHEMA.check_match(encrypted_key)

  # Does 'passphrase' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(passphrase)

  # Store and return the decrypted key object.
  key_object = None

  # Decrypt 'encrypted_key' so that the original key object is restored.
  # encrypt_key() generates an encrypted string of the key object using
  # AES-256-CTR-Mode, where 'password' is strengthened with PBKDF2-HMAC-SHA256.
  key_object = \
    securesystemslib.pyca_crypto_keys.decrypt_key(encrypted_key, passphrase)

  # The corresponding encrypt_key() encrypts and stores key objects in
  # non-metadata format (i.e., original format of key object argument to
  # encrypt_key()) prior to returning.

  return key_object