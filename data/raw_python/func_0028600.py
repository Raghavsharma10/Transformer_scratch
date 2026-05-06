def generate_and_write_rsa_keypair(filepath=None, bits=DEFAULT_RSA_KEY_BITS,
    password=None):
  """
  <Purpose>
    Generate an RSA key pair.  The public portion of the generated RSA key is
    saved to <'filepath'>.pub, whereas the private key portion is saved to
    <'filepath'>.  If no password is given, the user is prompted for one.  If
    the 'password' is an empty string, the private key is saved unencrypted to
    <'filepath'>.  If the filepath is not given, the KEYID is used as the
    filename and the keypair saved to the current working directory.

    The best available form of encryption, for a given key's backend, is used
    with pyca/cryptography.  According to their documentation, "it is a curated
    encryption choice and the algorithm may change over time."

  <Arguments>
    filepath:
      The public and private key files are saved to <filepath>.pub and
      <filepath>, respectively.  If the filepath is not given, the public and
      private keys are saved to the current working directory as <KEYID>.pub
      and <KEYID>.  KEYID is the generated key's KEYID.

    bits:
      The number of bits of the generated RSA key.

    password:
      The password to encrypt 'filepath'.  If None, the user is prompted for a
      password.  If an empty string is given, the private key is written to
      disk unencrypted.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

  <Side Effects>
    Writes key files to '<filepath>' and '<filepath>.pub'.

  <Returns>
    The 'filepath' of the written key.
  """

  # Does 'bits' have the correct format?
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.RSAKEYBITS_SCHEMA.check_match(bits)

  # Generate the public and private RSA keys.
  rsa_key = securesystemslib.keys.generate_rsa_key(bits)
  public = rsa_key['keyval']['public']
  private = rsa_key['keyval']['private']

  if not filepath:
    filepath = os.path.join(os.getcwd(), rsa_key['keyid'])

  else:
    logger.debug('The filepath has been specified.  Not using the key\'s'
        ' KEYID as the default filepath.')

  # Does 'filepath' have the correct format?
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

  # If the caller does not provide a password argument, prompt for one.
  if password is None: # pragma: no cover

    # It is safe to specify the full path of 'filepath' in the prompt and not
    # worry about leaking sensitive information about the key's location.
    # However, care should be taken when including the full path in exceptions
    # and log files.
    password = get_password('Enter a password for the encrypted RSA'
        ' key (' + Fore.RED + filepath + Fore.RESET + '): ',
        confirm=True)

  else:
    logger.debug('The password has been specified.  Not prompting for one')

  # Does 'password' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  # Encrypt the private key if 'password' is set.
  if len(password):
    private = securesystemslib.keys.create_rsa_encrypted_pem(private, password)

  else:
    logger.debug('An empty password was given.  Not encrypting the private key.')

  # If the parent directory of filepath does not exist,
  # create it (and all its parent directories, if necessary).
  securesystemslib.util.ensure_parent_dir(filepath)

  # Write the public key (i.e., 'public', which is in PEM format) to
  # '<filepath>.pub'.  (1) Create a temporary file, (2) write the contents of
  # the public key, and (3) move to final destination.
  file_object = securesystemslib.util.TempFile()
  file_object.write(public.encode('utf-8'))
  # The temporary file is closed after the final move.
  file_object.move(filepath + '.pub')

  # Write the private key in encrypted PEM format to '<filepath>'.
  # Unlike the public key file, the private key does not have a file
  # extension.
  file_object = securesystemslib.util.TempFile()
  file_object.write(private.encode('utf-8'))
  file_object.move(filepath)

  return filepath