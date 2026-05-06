def create_rsa_encrypted_pem(private_key, passphrase):
  """
  <Purpose>
    Return a string in PEM format (TraditionalOpenSSL), where the private part
    of the RSA key is encrypted using the best available encryption for a given
    key's backend. This is a curated (by cryptography.io) encryption choice and
    the algorithm may change over time.

    c.f. cryptography.io/en/latest/hazmat/primitives/asymmetric/serialization/
        #cryptography.hazmat.primitives.serialization.BestAvailableEncryption

    >>> public, private = generate_rsa_public_and_private(2048)
    >>> passphrase = 'secret'
    >>> encrypted_pem = create_rsa_encrypted_pem(private, passphrase)
    >>> securesystemslib.formats.PEMRSA_SCHEMA.matches(encrypted_pem)
    True

  <Arguments>
    private_key:
      The private key string in PEM format.

    passphrase:
      The passphrase, or password, to encrypt the private part of the RSA
      key.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
        formatted.

    securesystemslib.exceptions.CryptoError, if the passed RSA key cannot be
        deserialized by pyca cryptography.

    ValueError, if 'private_key' is unset.


  <Returns>
    A string in PEM format (TraditionalOpenSSL), where the private RSA key is
    encrypted. Conforms to 'securesystemslib.formats.PEMRSA_SCHEMA'.
  """

  # This check will ensure 'private_key' has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(private_key)

  # Does 'passphrase' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(passphrase)

  # 'private_key' may still be a NULL string after the
  # 'securesystemslib.formats.PEMRSA_SCHEMA' so we need an additional check
  if len(private_key):
    try:
      private_key = load_pem_private_key(private_key.encode('utf-8'),
          password=None, backend=default_backend())
    except ValueError:
      raise securesystemslib.exceptions.CryptoError('The private key'
          ' (in PEM format) could not be deserialized.')

  else:
    raise ValueError('The required private key is unset.')

  encrypted_pem = private_key.private_bytes(
      encoding=serialization.Encoding.PEM,
      format=serialization.PrivateFormat.TraditionalOpenSSL,
      encryption_algorithm=serialization.BestAvailableEncryption(
      passphrase.encode('utf-8')))

  return encrypted_pem.decode()