def generate_public_and_private(scheme='ecdsa-sha2-nistp256'):
  """
  <Purpose>
    Generate a pair of ECDSA public and private keys with one of the supported,
    external cryptography libraries.  The public and private keys returned
    conform to 'securesystemslib.formats.PEMECDSA_SCHEMA' and
    'securesystemslib.formats.PEMECDSA_SCHEMA', respectively.

    The public ECDSA public key has the PEM format:
    TODO: should we encrypt the private keys returned here?  Should the
    create_signature() accept encrypted keys?

    '-----BEGIN PUBLIC KEY-----

    ...

    '-----END PUBLIC KEY-----'



    The private ECDSA private key has the PEM format:

    '-----BEGIN EC PRIVATE KEY-----

    ...

    -----END EC PRIVATE KEY-----'

    >>> public, private = generate_public_and_private()
    >>> securesystemslib.formats.PEMECDSA_SCHEMA.matches(public)
    True
    >>> securesystemslib.formats.PEMECDSA_SCHEMA.matches(private)
    True

  <Arguments>
    scheme:
      A string indicating which algorithm to use for the generation of the
      public and private ECDSA keys.  'ecdsa-sha2-nistp256' is the only
      currently supported ECDSA algorithm, which is supported by OpenSSH and
      specified in RFC 5656 (https://tools.ietf.org/html/rfc5656).

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'algorithm' is improperly
    formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if 'scheme' is an
    unsupported algorithm.

  <Side Effects>
    None.

  <Returns>
    A (public, private) tuple that conform to
    'securesystemslib.formats.PEMECDSA_SCHEMA' and
    'securesystemslib.formats.PEMECDSA_SCHEMA', respectively.
  """

  # Does 'scheme' have the correct format?
  # Verify that 'scheme' is of the correct type, and that it's one of the
  # supported ECDSA .  It must conform to
  # 'securesystemslib.formats.ECDSA_SCHEME_SCHEMA'.  Raise
  # 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.ECDSA_SCHEME_SCHEMA.check_match(scheme)

  public_key = None
  private_key = None

  # An if-clause is strictly not needed, since 'ecdsa_sha2-nistp256' is the
  # only currently supported ECDSA signature scheme.  Nevertheness, include the
  # conditional statement to accomodate any schemes that might be added.
  if scheme == 'ecdsa-sha2-nistp256':
    private_key = ec.generate_private_key(ec.SECP256R1, default_backend())
    public_key = private_key.public_key()

  # The ECDSA_SCHEME_SCHEMA.check_match() above should have detected any
  # invalid 'scheme'.  This is a defensive check.
  else: #pragma: no cover
    raise securesystemslib.exceptions.UnsupportedAlgorithmError('An unsupported'
      ' scheme specified: ' + repr(scheme) + '.\n  Supported'
      ' algorithms: ' + repr(_SUPPORTED_ECDSA_SCHEMES))

  private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption())

  public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo)

  return public_pem.decode('utf-8'), private_pem.decode('utf-8')