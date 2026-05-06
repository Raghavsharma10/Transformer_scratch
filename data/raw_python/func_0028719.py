def create_rsa_encrypted_pem(private_key, passphrase):
  """
  <Purpose>
    Return a string in PEM format (TraditionalOpenSSL), where the private part
    of the RSA key is encrypted using the best available encryption for a given
    key's backend. This is a curated (by cryptography.io) encryption choice and
    the algorithm may change over time.

    c.f. cryptography.io/en/latest/hazmat/primitives/asymmetric/serialization/
        #cryptography.hazmat.primitives.serialization.BestAvailableEncryption

  >>> rsa_key = generate_rsa_key()
  >>> private = rsa_key['keyval']['private']
  >>> passphrase = 'secret'
  >>> encrypted_pem = create_rsa_encrypted_pem(private, passphrase)
  >>> securesystemslib.formats.PEMRSA_SCHEMA.matches(encrypted_pem)
  True

  <Arguments>
    private_key:
      The private key string in PEM format.

    passphrase:
      The passphrase, or password, to encrypt the private part of the RSA key.
      'passphrase' is not used directly as the encryption key, a stronger
      encryption key is derived from it.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.CryptoError, if an RSA key in encrypted PEM
    format cannot be created.

    TypeError, 'private_key' is unset.

  <Side Effects>
    None.

  <Returns>
    A string in PEM format, where the private RSA key is encrypted.
    Conforms to 'securesystemslib.formats.PEMRSA_SCHEMA'.
  """

  # Does 'private_key' have the correct format?
  # This check will ensure 'private_key' has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(private_key)

  # Does 'passphrase' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(passphrase)

  encrypted_pem = None

  # Generate the public and private RSA keys. A 2048-bit minimum is enforced by
  # create_rsa_encrypted_pem() via a
  # securesystemslib.formats.RSAKEYBITS_SCHEMA.check_match().
  encrypted_pem = securesystemslib.pyca_crypto_keys.create_rsa_encrypted_pem(
      private_key, passphrase)

  return encrypted_pem