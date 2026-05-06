def create_rsa_public_and_private_from_pem(pem, passphrase=None):
  """
  <Purpose>
    Generate public and private RSA keys from an optionally encrypted PEM.  The
    public and private keys returned conform to
    'securesystemslib.formats.PEMRSA_SCHEMA' and have the form:

    '-----BEGIN RSA PUBLIC KEY----- ... -----END RSA PUBLIC KEY-----'

    and

    '-----BEGIN RSA PRIVATE KEY----- ...-----END RSA PRIVATE KEY-----'

    The public and private keys are returned as strings in PEM format.

    In case the private key part of 'pem' is encrypted  pyca/cryptography's
    load_pem_private_key() method is passed passphrase.  In the default case
    here, pyca/cryptography will decrypt with a PBKDF1+MD5
    strengthened'passphrase', and 3DES with CBC mode for encryption/decryption.
    Alternatively, key data may be encrypted with AES-CTR-Mode and the
    passphrase strengthened with PBKDF2+SHA256, although this method is used
    only with TUF encrypted key files.

    >>> public, private = generate_rsa_public_and_private(2048)
    >>> passphrase = 'secret'
    >>> encrypted_pem = create_rsa_encrypted_pem(private, passphrase)
    >>> returned_public, returned_private = \
    create_rsa_public_and_private_from_pem(encrypted_pem, passphrase)
    >>> securesystemslib.formats.PEMRSA_SCHEMA.matches(returned_public)
    True
    >>> securesystemslib.formats.PEMRSA_SCHEMA.matches(returned_private)
    True
    >>> public == returned_public
    True
    >>> private == returned_private
    True

  <Arguments>
    pem:
      A byte string in PEM format, where the private key can be encrypted.
      It has the form:

      '-----BEGIN RSA PRIVATE KEY-----\n
      Proc-Type: 4,ENCRYPTED\nDEK-Info: DES-EDE3-CBC ...'

    passphrase: (optional)
      The passphrase, or password, to decrypt the private part of the RSA
      key.  'passphrase' is not directly used as the encryption key, instead
      it is used to derive a stronger symmetric key.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.CryptoError, if the public and private RSA keys
    cannot be generated from 'pem', or exported in PEM format.

  <Side Effects>
    pyca/cryptography's 'serialization.load_pem_private_key()' called to
    perform the actual conversion from an encrypted RSA private key to
    PEM format.

  <Returns>
    A (public, private) tuple containing the RSA keys in PEM format.
  """

  # Does 'encryped_pem' have the correct format?
  # This check will ensure 'pem' has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(pem)

  # If passed, does 'passphrase' have the correct format?
  if passphrase is not None:
    securesystemslib.formats.PASSWORD_SCHEMA.check_match(passphrase)
    passphrase = passphrase.encode('utf-8')

  # Generate a pyca/cryptography key object from 'pem'.  The generated
  # pyca/cryptography key contains the required export methods needed to
  # generate the PEM-formatted representations of the public and private RSA
  # key.
  try:
    private_key = load_pem_private_key(pem.encode('utf-8'),
        passphrase, backend=default_backend())

  # pyca/cryptography's expected exceptions for 'load_pem_private_key()':
  # ValueError: If the PEM data could not be decrypted.
  # (possibly because the passphrase is wrong)."
  # TypeError: If a password was given and the private key was not encrypted.
  # Or if the key was encrypted but no password was supplied.
  # UnsupportedAlgorithm: If the private key (or if the key is encrypted with
  # an unsupported symmetric cipher) is not supported by the backend.
  except (ValueError, TypeError, cryptography.exceptions.UnsupportedAlgorithm) as e:
    # Raise 'securesystemslib.exceptions.CryptoError' and pyca/cryptography's
    # exception message.  Avoid propogating pyca/cryptography's exception trace
    # to avoid revealing sensitive error.
    raise securesystemslib.exceptions.CryptoError('RSA (public, private) tuple'
      ' cannot be generated from the encrypted PEM string: ' + str(e))

  # Export the public and private halves of the pyca/cryptography RSA key
  # object.  The (public, private) tuple returned contains the public and
  # private RSA keys in PEM format, as strings.
  # Extract the public & private halves of the RSA key and generate their
  # PEM-formatted representations.  Return the key pair as a (public, private)
  # tuple, where each RSA is a string in PEM format.
  private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
      format=serialization.PrivateFormat.TraditionalOpenSSL,
      encryption_algorithm=serialization.NoEncryption())

  # Need to generate the public key from the private one before serializing
  # to PEM format.
  public_key = private_key.public_key()
  public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM,
      format=serialization.PublicFormat.SubjectPublicKeyInfo)

  return public_pem.decode(), private_pem.decode()