def generate_and_write_ed25519_keypair(filepath=None, password=None):
  """
  <Purpose>
    Generate an Ed25519 keypair, where the encrypted key (using 'password' as
    the passphrase) is saved to <'filepath'>.  The public key portion of the
    generated Ed25519 key is saved to <'filepath'>.pub.  If the filepath is not
    given, the KEYID is used as the filename and the keypair saved to the
    current working directory.

    The private key is encrypted according to 'cryptography's approach:
    "Encrypt using the best available encryption for a given key's backend.
    This is a curated encryption choice and the algorithm may change over
    time."

  <Arguments>
    filepath:
      The public and private key files are saved to <filepath>.pub and
      <filepath>, respectively.  If the filepath is not given, the public and
      private keys are saved to the current working directory as <KEYID>.pub
      and <KEYID>.  KEYID is the generated key's KEYID.

    password:
      The password, or passphrase, to encrypt the private portion of the
      generated Ed25519 key.  A symmetric encryption key is derived from
      'password', so it is not directly used.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.CryptoError, if 'filepath' cannot be encrypted.

  <Side Effects>
    Writes key files to '<filepath>' and '<filepath>.pub'.

  <Returns>
    The 'filepath' of the written key.
  """

  # Generate a new Ed25519 key object.
  ed25519_key = securesystemslib.keys.generate_ed25519_key()

  if not filepath:
    filepath = os.path.join(os.getcwd(), ed25519_key['keyid'])

  else:
    logger.debug('The filepath has been specified.  Not using the key\'s'
        ' KEYID as the default filepath.')

  # Does 'filepath' have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

  # If the caller does not provide a password argument, prompt for one.
  if password is None: # pragma: no cover

    # It is safe to specify the full path of 'filepath' in the prompt and not
    # worry about leaking sensitive information about the key's location.
    # However, care should be taken when including the full path in exceptions
    # and log files.
    password = get_password('Enter a password for the Ed25519'
        ' key (' + Fore.RED + filepath + Fore.RESET + '): ',
        confirm=True)

  else:
    logger.debug('The password has been specified. Not prompting for one.')

  # Does 'password' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  # If the parent directory of filepath does not exist,
  # create it (and all its parent directories, if necessary).
  securesystemslib.util.ensure_parent_dir(filepath)

  # Create a temporary file, write the contents of the public key, and move
  # to final destination.
  file_object = securesystemslib.util.TempFile()

  # Generate the ed25519 public key file contents in metadata format (i.e.,
  # does not include the keyid portion).
  keytype = ed25519_key['keytype']
  keyval = ed25519_key['keyval']
  scheme = ed25519_key['scheme']
  ed25519key_metadata_format = securesystemslib.keys.format_keyval_to_metadata(
      keytype, scheme, keyval, private=False)

  file_object.write(json.dumps(ed25519key_metadata_format).encode('utf-8'))

  # Write the public key (i.e., 'public', which is in PEM format) to
  # '<filepath>.pub'.  (1) Create a temporary file, (2) write the contents of
  # the public key, and (3) move to final destination.
  # The temporary file is closed after the final move.
  file_object.move(filepath + '.pub')

  # Write the encrypted key string, conformant to
  # 'securesystemslib.formats.ENCRYPTEDKEY_SCHEMA', to '<filepath>'.
  file_object = securesystemslib.util.TempFile()

  # Encrypt the private key if 'password' is set.
  if len(password):
    ed25519_key = securesystemslib.keys.encrypt_key(ed25519_key, password)

  else:
    logger.debug('An empty password was given. '
                 'Not encrypting the private key.')
    ed25519_key = json.dumps(ed25519_key)

  # Raise 'securesystemslib.exceptions.CryptoError' if 'ed25519_key' cannot be
  # encrypted.
  file_object.write(ed25519_key.encode('utf-8'))
  file_object.move(filepath)

  return filepath