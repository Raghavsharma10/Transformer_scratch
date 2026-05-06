def create_signature(public_key, private_key, data, scheme='ecdsa-sha2-nistp256'):
  """
  <Purpose>
    Return a (signature, scheme) tuple.

    >>> requested_scheme = 'ecdsa-sha2-nistp256'
    >>> public, private = generate_public_and_private(requested_scheme)
    >>> data = b'The quick brown fox jumps over the lazy dog'
    >>> signature, scheme = create_signature(public, private, data, requested_scheme)
    >>> securesystemslib.formats.ECDSASIGNATURE_SCHEMA.matches(signature)
    True
    >>> requested_scheme == scheme
    True

  <Arguments>
    public:
      The ECDSA public key in PEM format.

    private:
      The ECDSA private key in PEM format.

    data:
      Byte data used by create_signature() to generate the signature returned.

    scheme:
      The signature scheme used to generate the signature.  For example:
      'ecdsa-sha2-nistp256'.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.CryptoError, if a signature cannot be created.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if 'scheme' is not
    one of the supported signature schemes.

  <Side Effects>
    None.

  <Returns>
    A signature dictionary conformat to
    'securesystemslib.format.SIGNATURE_SCHEMA'.  ECDSA signatures are XX bytes,
    however, the hexlified signature is stored in the dictionary returned.
  """

  # Do 'public_key' and 'private_key' have the correct format?
  # This check will ensure that the arguments conform to
  # 'securesystemslib.formats.PEMECDSA_SCHEMA'.  Raise
  # 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMECDSA_SCHEMA.check_match(public_key)

  # Is 'private_key' properly formatted?
  securesystemslib.formats.PEMECDSA_SCHEMA.check_match(private_key)

  # Is 'scheme' properly formatted?
  securesystemslib.formats.ECDSA_SCHEME_SCHEMA.check_match(scheme)

  # 'ecdsa-sha2-nistp256' is the only currently supported ECDSA scheme, so this
  # if-clause isn't strictly needed.  Nevertheless, the conditional statement
  # is included to accommodate multiple schemes that can potentially be added
  # in the future.
  if scheme == 'ecdsa-sha2-nistp256':
    try:
      private_key = load_pem_private_key(private_key.encode('utf-8'),
        password=None, backend=default_backend())

      signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))

    except TypeError as e:
      raise securesystemslib.exceptions.CryptoError('Could not create'
        ' signature: ' + str(e))

  # A defensive check for an invalid 'scheme'.  The
  # ECDSA_SCHEME_SCHEMA.check_match() above should have already validated it.
  else: #pragma: no cover
    raise securesystemslib.exceptions.UnsupportedAlgorithmError('Unsupported'
      ' signature scheme is specified: ' + repr(scheme))

  return signature, scheme