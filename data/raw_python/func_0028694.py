def verify_rsa_signature(signature, signature_scheme, public_key, data):
  """
  <Purpose>
    Determine whether the corresponding private key of 'public_key' produced
    'signature'.  verify_signature() will use the public key, signature scheme,
    and 'data' to complete the verification.

    >>> public, private = generate_rsa_public_and_private(2048)
    >>> data = b'The quick brown fox jumps over the lazy dog'
    >>> scheme = 'rsassa-pss-sha256'
    >>> signature, scheme = create_rsa_signature(private, data, scheme)
    >>> verify_rsa_signature(signature, scheme, public, data)
    True
    >>> verify_rsa_signature(signature, scheme, public, b'bad_data')
    False

  <Arguments>
    signature:
      A signature, as a string.  This is the signature returned
      by create_rsa_signature().

    signature_scheme:
      A string that indicates the signature scheme used to generate
      'signature'.  'rsassa-pss-sha256' is currently supported.

    public_key:
      The RSA public key, a string in PEM format.

    data:
      Data used by securesystemslib.keys.create_signature() to generate
      'signature'.  'data' (a string) is needed here to verify 'signature'.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'signature',
    'signature_scheme', 'public_key', or 'data' are improperly formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if the signature
    scheme used by 'signature' is not one supported by
    securesystemslib.keys.create_signature().

    securesystemslib.exceptions.CryptoError, if the private key cannot be
    decoded or its key type is unsupported.

  <Side Effects>
    pyca/cryptography's RSAPublicKey.verifier() called to do the actual
    verification.

   <Returns>
    Boolean.  True if the signature is valid, False otherwise.
  """

  # Does 'public_key' have the correct format?
  # This check will ensure 'public_key' conforms to
  # 'securesystemslib.formats.PEMRSA_SCHEMA'.  Raise
  # 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(public_key)

  # Does 'signature_scheme' have the correct format?
  securesystemslib.formats.RSA_SCHEME_SCHEMA.check_match(signature_scheme)

  # Does 'signature' have the correct format?
  securesystemslib.formats.PYCACRYPTOSIGNATURE_SCHEMA.check_match(signature)

  # What about 'data'?
  securesystemslib.formats.DATA_SCHEMA.check_match(data)

  # Verify whether the private key of 'public_key' produced 'signature'.
  # Before returning the 'valid_signature' Boolean result, ensure 'RSASSA-PSS'
  # was used as the signature scheme.
  valid_signature = False

  # Verify the RSASSA-PSS signature with pyca/cryptography.
  try:
    public_key_object = serialization.load_pem_public_key(public_key.encode('utf-8'),
        backend=default_backend())

    # verify() raises 'cryptography.exceptions.InvalidSignature' if the
    # signature is invalid. 'salt_length' is set to the digest size of the
    # hashing algorithm.
    try:
      public_key_object.verify(signature, data,
          padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
          salt_length=hashes.SHA256().digest_size),
          hashes.SHA256())
      return True

    except cryptography.exceptions.InvalidSignature:
      return False

  # Raised by load_pem_public_key().
  except (ValueError, cryptography.exceptions.UnsupportedAlgorithm) as e:
    raise securesystemslib.exceptions.CryptoError('The PEM could not be'
      ' decoded successfully, or contained an unsupported key type: ' + str(e))