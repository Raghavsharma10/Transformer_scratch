def create_ecdsa_public_and_private_from_pem(pem, password=None):
  """
  <Purpose>
    Create public and private ECDSA keys from a private 'pem'.  The public and
    private keys are strings in PEM format:

    public: '-----BEGIN PUBLIC KEY----- ... -----END PUBLIC KEY-----',
    private: '-----BEGIN EC PRIVATE KEY----- ... -----END EC PRIVATE KEY-----'}}

    >>> junk, private = generate_public_and_private()
    >>> public, private = create_ecdsa_public_and_private_from_pem(private)
    >>> securesystemslib.formats.PEMECDSA_SCHEMA.matches(public)
    True
    >>> securesystemslib.formats.PEMECDSA_SCHEMA.matches(private)
    True
    >>> passphrase = 'secret'
    >>> encrypted_pem = create_ecdsa_encrypted_pem(private, passphrase)
    >>> public, private = create_ecdsa_public_and_private_from_pem(encrypted_pem, passphrase)
    >>> securesystemslib.formats.PEMECDSA_SCHEMA.matches(public)
    True
    >>> securesystemslib.formats.PEMECDSA_SCHEMA.matches(private)
    True

  <Arguments>
    pem:
      A string in PEM format.  The private key is extracted and returned in
      an ecdsakey object.

    password: (optional)
      The password, or passphrase, to decrypt the private part of the ECDSA key
      if it is encrypted.  'password' is not used directly as the encryption
      key, a stronger encryption key is derived from it.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if the ECDSA key
    pair could not be extracted, possibly due to an unsupported algorithm.

  <Side Effects>
    None.

  <Returns>
    A dictionary containing the ECDSA keys and other identifying information.
    Conforms to 'securesystemslib.formats.ECDSAKEY_SCHEMA'.
  """

  # Does 'pem' have the correct format?
  # This check will ensure 'pem' conforms to
  # 'securesystemslib.formats.ECDSARSA_SCHEMA'.
  securesystemslib.formats.PEMECDSA_SCHEMA.check_match(pem)

  if password is not None:
    securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)
    password = password.encode('utf-8')

  else:
    logger.debug('The password/passphrase is unset.  The PEM is expected'
      ' to be unencrypted.')

  public = None
  private = None

  # Generate the public and private ECDSA keys.  The pyca/cryptography library
  # performs the actual import operation.
  try:
    private = load_pem_private_key(pem.encode('utf-8'), password=password,
      backend=default_backend())

  except (ValueError, cryptography.exceptions.UnsupportedAlgorithm) as e:
    raise securesystemslib.exceptions.CryptoError('Could not import private'
      ' PEM.\n' + str(e))

  public = private.public_key()

  # Serialize public and private keys to PEM format.
  private = private.private_bytes(encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption())

  public = public.public_bytes(encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo)

  return public.decode('utf-8'), private.decode('utf-8')