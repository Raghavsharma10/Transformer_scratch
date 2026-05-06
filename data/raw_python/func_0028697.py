def encrypt_key(key_object, password):
  """
  <Purpose>
    Return a string containing 'key_object' in encrypted form. Encrypted
    strings may be safely saved to a file.  The corresponding decrypt_key()
    function can be applied to the encrypted string to restore the original key
    object.  'key_object' is a TUF key (e.g., RSAKEY_SCHEMA,
    ED25519KEY_SCHEMA).  This function calls the pyca/cryptography library to
    perform the encryption and derive a suitable encryption key.

    Whereas an encrypted PEM file uses the Triple Data Encryption Algorithm
    (3DES), the Cipher-block chaining (CBC) mode of operation, and the Password
    Based Key Derivation Function 1 (PBKF1) + MD5 to strengthen 'password',
    encrypted TUF keys use AES-256-CTR-Mode and passwords strengthened with
    PBKDF2-HMAC-SHA256 (100K iterations by default, but may be overriden in
    'settings.PBKDF2_ITERATIONS' by the user).

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
    >>> securesystemslib.formats.ENCRYPTEDKEY_SCHEMA.matches(encrypted_key.encode('utf-8'))
    True

  <Arguments>
    key_object:
      The TUF key object that should contain the private portion of the ED25519
      key.

    password:
      The password, or passphrase, to encrypt the private part of the RSA
      key.  'password' is not used directly as the encryption key, a stronger
      encryption key is derived from it.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if any of the arguments are
    improperly formatted or 'key_object' does not contain the private portion
    of the key.

    securesystemslib.exceptions.CryptoError, if an Ed25519 key in encrypted TUF
    format cannot be created.

  <Side Effects>
    pyca/Cryptography cryptographic operations called to perform the actual
    encryption of 'key_object'.  'password' used to derive a suitable
    encryption key.

  <Returns>
    An encrypted string in 'securesystemslib.formats.ENCRYPTEDKEY_SCHEMA' format.
  """

  # Do the arguments have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.ANYKEY_SCHEMA.check_match(key_object)

  # Does 'password' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  # Ensure the private portion of the key is included in 'key_object'.
  if 'private' not in key_object['keyval'] or not key_object['keyval']['private']:
    raise securesystemslib.exceptions.FormatError('Key object does not contain'
      ' a private part.')

  # Derive a key (i.e., an appropriate encryption key and not the
  # user's password) from the given 'password'.  Strengthen 'password' with
  # PBKDF2-HMAC-SHA256 (100K iterations by default, but may be overriden in
  # 'settings.PBKDF2_ITERATIONS' by the user).
  salt, iterations, derived_key = _generate_derived_key(password)

  # Store the derived key info in a dictionary, the object expected
  # by the non-public _encrypt() routine.
  derived_key_information = {'salt': salt, 'iterations': iterations,
                             'derived_key': derived_key}

  # Convert the key object to json string format and encrypt it with the
  # derived key.
  encrypted_key = _encrypt(json.dumps(key_object), derived_key_information)

  return encrypted_key