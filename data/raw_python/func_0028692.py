def generate_rsa_public_and_private(bits=_DEFAULT_RSA_KEY_BITS):
  """
  <Purpose>
    Generate public and private RSA keys with modulus length 'bits'.  The
    public and private keys returned conform to
    'securesystemslib.formats.PEMRSA_SCHEMA' and have the form:

    '-----BEGIN RSA PUBLIC KEY----- ...'

    or

    '-----BEGIN RSA PRIVATE KEY----- ...'

    The public and private keys are returned as strings in PEM format.

    'generate_rsa_public_and_private()' enforces a minimum key size of 2048
    bits.  If 'bits' is unspecified, a 3072-bit RSA key is generated, which is
    the key size recommended by TUF.

    >>> public, private = generate_rsa_public_and_private(2048)
    >>> securesystemslib.formats.PEMRSA_SCHEMA.matches(public)
    True
    >>> securesystemslib.formats.PEMRSA_SCHEMA.matches(private)
    True

  <Arguments>
    bits:
      The key size, or key length, of the RSA key.  'bits' must be 2048, or
      greater.  'bits' defaults to 3072 if not specified.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'bits' does not contain the
    correct format.

  <Side Effects>
    The RSA keys are generated from pyca/cryptography's
    rsa.generate_private_key() function.

  <Returns>
    A (public, private) tuple containing the RSA keys in PEM format.
  """

  # Does 'bits' have the correct format?
  # This check will ensure 'bits' conforms to
  # 'securesystemslib.formats.RSAKEYBITS_SCHEMA'.  'bits' must be an integer
  # object, with a minimum value of 2048.  Raise
  # 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.RSAKEYBITS_SCHEMA.check_match(bits)

  # Generate the public and private RSA keys.  The pyca/cryptography 'rsa'
  # module performs the actual key generation.  The 'bits' argument is used,
  # and a 2048-bit minimum is enforced by
  # securesystemslib.formats.RSAKEYBITS_SCHEMA.check_match().
  private_key = rsa.generate_private_key(public_exponent=65537, key_size=bits,
      backend=default_backend())

  # Extract the public & private halves of the RSA key and generate their
  # PEM-formatted representations.  Return the key pair as a (public, private)
  # tuple, where each RSA is a string in PEM format.
  private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
      format=serialization.PrivateFormat.TraditionalOpenSSL,
      encryption_algorithm=serialization.NoEncryption())

  # Need to generate the public pem from the private key before serialization
  # to PEM.
  public_key = private_key.public_key()
  public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM,
      format=serialization.PublicFormat.SubjectPublicKeyInfo)

  return public_pem.decode('utf-8'), private_pem.decode('utf-8')