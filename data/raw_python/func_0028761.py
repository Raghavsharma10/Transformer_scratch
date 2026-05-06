def verify_signature(public_key, scheme, signature, data, use_pynacl=False):
  """
  <Purpose>
    Determine whether the private key corresponding to 'public_key' produced
    'signature'.  verify_signature() will use the public key, the 'scheme' and
    'sig', and 'data' arguments to complete the verification.

    >>> public, private = generate_public_and_private()
    >>> data = b'The quick brown fox jumps over the lazy dog'
    >>> scheme = 'ed25519'
    >>> signature, scheme = \
        create_signature(public, private, data, scheme)
    >>> verify_signature(public, scheme, signature, data, use_pynacl=False)
    True
    >>> verify_signature(public, scheme, signature, data, use_pynacl=True)
    True
    >>> bad_data = b'The sly brown fox jumps over the lazy dog'
    >>> bad_signature, scheme = \
        create_signature(public, private, bad_data, scheme)
    >>> verify_signature(public, scheme, bad_signature, data, use_pynacl=False)
    False

  <Arguments>
    public_key:
      The public key is a 32-byte string.

    scheme:
      'ed25519' signature scheme used by either the pure python
      implementation (i.e., ed25519.py) or PyNacl (i.e., 'nacl').

    signature:
      The signature is a 64-byte string.

    data:
      Data object used by securesystemslib.ed25519_keys.create_signature() to
      generate 'signature'.  'data' is needed here to verify the signature.

    use_pynacl:
      True, if the ed25519 signature should be verified by PyNaCl.  False,
      if the signature should be verified with the pure Python implementation
      of ed25519 (slower).

  <Exceptions>
    securesystemslib.exceptions.UnsupportedAlgorithmError.  Raised if the
    signature scheme 'scheme' is not one supported by
    securesystemslib.ed25519_keys.create_signature().

    securesystemslib.exceptions.FormatError. Raised if the arguments are
    improperly formatted.

  <Side Effects>
    securesystemslib._vendor.ed25519.ed25519.checkvalid() called to do the
    actual verification.  nacl.signing.VerifyKey.verify() called if
    'use_pynacl' is True.

  <Returns>
    Boolean.  True if the signature is valid, False otherwise.
  """

  # Does 'public_key' have the correct format?
  # This check will ensure 'public_key' conforms to
  # 'securesystemslib.formats.ED25519PUBLIC_SCHEMA', which must have length 32
  # bytes.  Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.ED25519PUBLIC_SCHEMA.check_match(public_key)

  # Is 'scheme' properly formatted?
  securesystemslib.formats.ED25519_SIG_SCHEMA.check_match(scheme)

  # Is 'signature' properly formatted?
  securesystemslib.formats.ED25519SIGNATURE_SCHEMA.check_match(signature)

  # Is 'use_pynacl' properly formatted?
  securesystemslib.formats.BOOLEAN_SCHEMA.check_match(use_pynacl)

  # Verify 'signature'.  Before returning the Boolean result, ensure 'ed25519'
  # was used as the signature scheme.  Raise
  # 'securesystemslib.exceptions.UnsupportedLibraryError' if 'use_pynacl' is
  # True but 'nacl' is unavailable.
  public = public_key
  valid_signature = False

  if scheme in _SUPPORTED_ED25519_SIGNING_SCHEMES:
    if use_pynacl:
      try:
        nacl_verify_key = nacl.signing.VerifyKey(public)
        nacl_message = nacl_verify_key.verify(data, signature)
        valid_signature = True

      # The unit tests expect PyNaCl to be installed.
      except NameError: # pragma: no cover
        raise securesystemslib.exceptions.UnsupportedLibraryError('The PyNaCl'
            ' library and/or its dependencies unavailable.')

      except nacl.exceptions.BadSignatureError:
        pass

    # Verify 'ed25519' signature with the pure Python implementation.
    else:
      try:
        securesystemslib._vendor.ed25519.ed25519.checkvalid(signature,
            data, public)
        valid_signature = True

      # The pure Python implementation raises 'Exception' if 'signature' is
      # invalid.
      except Exception as e:
        pass

  # This is a defensive check for a valid 'scheme', which should have already
  # been validated in the ED25519_SIG_SCHEMA.check_match(scheme) above.
  else: #pragma: no cover
    message = 'Unsupported ed25519 signature scheme: ' + repr(scheme) + '.\n' + \
      'Supported schemes: ' + repr(_SUPPORTED_ED25519_SIGNING_SCHEMES) + '.'
    raise securesystemslib.exceptions.UnsupportedAlgorithmError(message)

  return valid_signature