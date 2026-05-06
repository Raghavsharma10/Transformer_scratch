def decrypt_key(encrypted_key, password):
  """
  <Purpose>
    Return a string containing 'encrypted_key' in non-encrypted form.
    The decrypt_key() function can be applied to the encrypted string to restore
    the original key object, a TUF key (e.g., RSAKEY_SCHEMA, ED25519KEY_SCHEMA).
    This function calls the appropriate cryptography module (i.e.,
    pyca_crypto_keys.py) to perform the decryption.

    Encrypted TUF keys use AES-256-CTR-Mode and passwords strengthened with
    PBKDF2-HMAC-SHA256 (100K iterations be default, but may be overriden in
    'settings.py' by the user).

    http://en.wikipedia.org/wiki/Advanced_Encryption_Standard
    http://en.wikipedia.org/wiki/CTR_mode#Counter_.28CTR.29
    https://en.wikipedia.org/wiki/PBKDF2

    >>> ed25519_key = {'keytype': 'ed25519', \
                       'scheme': 'ed25519', \
                       'keyid': \
          'd62247f817883f593cf6c66a5a55292488d457bcf638ae03207dbbba9dbe457d', \
                       'keyval': {'public': \
          '74addb5ad544a4306b34741bc1175a3613a8d7dc69ff64724243efdec0e301ad', \
                                  'private': \
          '1f26964cc8d4f7ee5f3c5da2fbb7ab35811169573ac367b860a537e47789f8c4'}}
    >>> passphrase = 'secret'
    >>> encrypted_key = encrypt_key(ed25519_key, passphrase)
    >>> decrypted_key = decrypt_key(encrypted_key.encode('utf-8'), passphrase)
    >>> securesystemslib.formats.ED25519KEY_SCHEMA.matches(decrypted_key)
    True
    >>> decrypted_key == ed25519_key
    True

  <Arguments>
    encrypted_key:
      An encrypted TUF key (additional data is also included, such as salt,
      number of password iterations used for the derived encryption key, etc)
      of the form 'securesystemslib.formats.ENCRYPTEDKEY_SCHEMA'.
      'encrypted_key' should have been generated with encrypted_key().

    password:
      The password, or passphrase, to encrypt the private part of the RSA
      key.  'password' is not used directly as the encryption key, a stronger
      encryption key is derived from it.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.CryptoError, if a TUF key cannot be decrypted
    from 'encrypted_key'.

    securesystemslib.exceptions.Error, if a valid TUF key object is not found in
    'encrypted_key'.

  <Side Effects>
    The pyca/cryptography is library called to perform the actual decryption
    of 'encrypted_key'.  The key derivation data stored in 'encrypted_key' is
    used to re-derive the encryption/decryption key.

  <Returns>
    The decrypted key object in 'securesystemslib.formats.ANYKEY_SCHEMA' format.
  """

  # Do the arguments have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.ENCRYPTEDKEY_SCHEMA.check_match(encrypted_key)

  # Does 'password' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  # Decrypt 'encrypted_key', using 'password' (and additional key derivation
  # data like salts and password iterations) to re-derive the decryption key.
  json_data = _decrypt(encrypted_key, password)

  # Raise 'securesystemslib.exceptions.Error' if 'json_data' cannot be
  # deserialized to a valid 'securesystemslib.formats.ANYKEY_SCHEMA' key
  # object.
  key_object = securesystemslib.util.load_json_string(json_data.decode())

  return key_object