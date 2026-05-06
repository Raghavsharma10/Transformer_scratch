def create_rsa_signature(private_key, data, scheme='rsassa-pss-sha256'):
  """
  <Purpose>
    Generate a 'scheme' signature.  The signature, and the signature scheme
    used, is returned as a (signature, scheme) tuple.

    The signing process will use 'private_key' to generate the signature of
    'data'.

    RFC3447 - RSASSA-PSS
    http://www.ietf.org/rfc/rfc3447.txt

    >>> public, private = generate_rsa_public_and_private(2048)
    >>> data = 'The quick brown fox jumps over the lazy dog'.encode('utf-8')
    >>> scheme = 'rsassa-pss-sha256'
    >>> signature, scheme = create_rsa_signature(private, data, scheme)
    >>> securesystemslib.formats.NAME_SCHEMA.matches(scheme)
    True
    >>> scheme == 'rsassa-pss-sha256'
    True
    >>> securesystemslib.formats.PYCACRYPTOSIGNATURE_SCHEMA.matches(signature)
    True

  <Arguments>
    private_key:
      The private RSA key, a string in PEM format.

    data:
      Data (string) used by create_rsa_signature() to generate the signature.

    scheme:
      The signature scheme used to generate the signature.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'private_key' is improperly
    formatted.

    ValueError, if 'private_key' is unset.

    securesystemslib.exceptions.CryptoError, if the signature cannot be
    generated.

  <Side Effects>
    pyca/cryptography's 'RSAPrivateKey.signer()' called to generate the
    signature.

  <Returns>
    A (signature, scheme) tuple, where the signature is a string and the scheme
    is one of the supported RSA signature schemes. For example:
    'rsassa-pss-sha256'.
  """

  # Does the arguments have the correct format?
  # If not, raise 'securesystemslib.exceptions.FormatError' if any of the
  # checks fail.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(private_key)
  securesystemslib.formats.DATA_SCHEMA.check_match(data)
  securesystemslib.formats.RSA_SCHEME_SCHEMA.check_match(scheme)

  # Signing 'data' requires a private key.  'rsassa-pss-sha256' is the only
  # currently supported signature scheme.
  signature = None

  # Verify the signature, but only if the private key has been set.  The
  # private key is a NULL string if unset.  Although it may be clearer to
  # explicitly check that 'private_key' is not '', we can/should check for a
  # value and not compare identities with the 'is' keyword.  Up to this point
  # 'private_key' has variable size and can be an empty string.
  if len(private_key):

    # An if-clause isn't strictly needed here, since 'rsasssa-pss-sha256' is
    # the only currently supported RSA scheme.  Nevertheless, include the
    # conditional statement to accomodate future schemes that might be added.
    if scheme == 'rsassa-pss-sha256':
      # Generate an RSSA-PSS signature.  Raise
      # 'securesystemslib.exceptions.CryptoError' for any of the expected
      # exceptions raised by pyca/cryptography.
      try:
        # 'private_key' (in PEM format) must first be converted to a
        # pyca/cryptography private key object before a signature can be
        # generated.
        private_key_object = load_pem_private_key(private_key.encode('utf-8'),
            password=None, backend=default_backend())

        signature = private_key_object.sign(
            data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256().digest_size), hashes.SHA256())

      # If the PEM data could not be decrypted, or if its structure could not
      # be decoded successfully.
      except ValueError:
        raise securesystemslib.exceptions.CryptoError('The private key'
          ' (in PEM format) could not be deserialized.')

      # 'TypeError' is raised if a password was given and the private key was
      # not encrypted, or if the key was encrypted but no password was
      # supplied.  Note: A passphrase or password is not used when generating
      # 'private_key', since it should not be encrypted.
      except TypeError:
        raise securesystemslib.exceptions.CryptoError('The private key was'
          ' unexpectedly encrypted.')

      # 'cryptography.exceptions.UnsupportedAlgorithm' is raised if the
      # serialized key is of a type that is not supported by the backend, or if
      # the key is encrypted with a symmetric cipher that is not supported by
      # the backend.
      except cryptography.exceptions.UnsupportedAlgorithm: #pragma: no cover
        raise securesystemslib.exceptions.CryptoError('The private key is'
          ' encrypted with an unsupported algorithm.')

    # The RSA_SCHEME_SCHEMA.check_match() above should have validated 'scheme'.
    # This is a defensive check check..
    else: #pragma: no cover
      raise securesystemslib.exceptions.UnsupportedAlgorithmError('Unsupported'
        ' signature scheme is specified: ' + repr(scheme))

  else:
    raise ValueError('The required private key is unset.')

  return signature, scheme