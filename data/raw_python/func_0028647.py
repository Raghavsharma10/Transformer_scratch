def create_ecdsa_encrypted_pem(private_pem, passphrase):
  """
  <Purpose>
    Return a string in PEM format, where the private part of the ECDSA key is
    encrypted. The private part of the ECDSA key is encrypted as done by
    pyca/cryptography: "Encrypt using the best available encryption for a given
    key's backend. This is a curated encryption choice and the algorithm may
    change over time."

    >>> junk, private = generate_public_and_private()
    >>> passphrase = 'secret'
    >>> encrypted_pem = create_ecdsa_encrypted_pem(private, passphrase)
    >>> securesystemslib.formats.PEMECDSA_SCHEMA.matches(encrypted_pem)
    True

  <Arguments>
    private_pem:
    The private ECDSA key string in PEM format.

    passphrase:
    The passphrase, or password, to encrypt the private part of the ECDSA
    key. 'passphrase' is not used directly as the encryption key, a stronger
    encryption key is derived from it.

    <Exceptions>
      securesystemslib.exceptions.FormatError, if the arguments are improperly
      formatted.

      securesystemslib.exceptions.CryptoError, if an ECDSA key in encrypted PEM
      format cannot be created.

  <Side Effects>
    None.

  <Returns>
    A string in PEM format, where the private RSA portion is encrypted.
    Conforms to 'securesystemslib.formats.PEMECDSA_SCHEMA'.
  """

  # Does 'private_key' have the correct format?
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(private_pem)

  # Does 'passphrase' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(passphrase)

  encrypted_pem = None

  private = load_pem_private_key(private_pem.encode('utf-8'), password=None,
    backend=default_backend())

  encrypted_private_pem = \
    private.private_bytes(encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.BestAvailableEncryption(passphrase.encode('utf-8')))

  return encrypted_private_pem