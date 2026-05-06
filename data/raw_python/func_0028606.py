def generate_and_write_ecdsa_keypair(filepath=None, password=None):
  """
  <Purpose>
    Generate an ECDSA keypair, where the encrypted key (using 'password' as the
    passphrase) is saved to <'filepath'>.  The public key portion of the
    generated ECDSA key is saved to <'filepath'>.pub.  If the filepath is not
    given, the KEYID is used as the filename and the keypair saved to the
    current working directory.

    The 'cryptography' library is currently supported.  The private key is
    encrypted according to 'cryptography's approach: "Encrypt using the best
    available encryption for a given key's backend. This is a curated
    encryption choice and the algorithm may change over time."

  <Arguments>
    filepath:
      The public and private key files are saved to <filepath>.pub and
      <filepath>, respectively.  If the filepath is not given, the public and
      private keys are saved to the current working directory as <KEYID>.pub
      and <KEYID>.  KEYID is the generated key's KEYID.

    password:
      The password, or passphrase, to encrypt the private portion of the
      generated ECDSA key.  A symmetric encryption key is derived from
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

  # Generate a new ECDSA key object.  The 'cryptography' library is currently
  # supported and performs the actual cryptographic operations.
  ecdsa_key = securesystemslib.keys.generate_ecdsa_key()

  if not filepath:
    filepath = os.path.join(os.getcwd(), ecdsa_key['keyid'])

  else:
    logger.debug('The filepath has been specified.  Not using the key\'s'
        ' KEYID as the default filepath.')

  # Does 'filepath' have the correct format?
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

  # If the caller does not provide a password argument, prompt for one.
  if password is None: # pragma: no cover

    # It is safe to specify the full path of 'filepath' in the prompt and not
    # worry about leaking sensitive information about the key's location.
    # However, care should be taken when including the full path in exceptions
    # and log files.
    password = get_password('Enter a password for the ECDSA'
        ' key (' + Fore.RED + filepath + Fore.RESET + '): ',
        confirm=True)

  else:
    logger.debug('The password has been specified.  Not prompting for one')

  # Does 'password' have the correct format?
  securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  # If the parent directory of filepath does not exist,
  # create it (and all its parent directories, if necessary).
  securesystemslib.util.ensure_parent_dir(filepath)

  # Create a temporary file, write the contents of the public key, and move
  # to final destination.
  file_object = securesystemslib.util.TempFile()

  # Generate the ECDSA public key file contents in metadata format (i.e., does
  # not include the keyid portion).
  keytype = ecdsa_key['keytype']
  keyval = ecdsa_key['keyval']
  scheme = ecdsa_key['scheme']
  ecdsakey_metadata_format = securesystemslib.keys.format_keyval_to_metadata(
      keytype, scheme, keyval, private=False)

  file_object.write(json.dumps(ecdsakey_metadata_format).encode('utf-8'))

  # Write the public key (i.e., 'public', which is in PEM format) to
  # '<filepath>.pub'.  (1) Create a temporary file, (2) write the contents of
  # the public key, and (3) move to final destination.
  file_object.move(filepath + '.pub')

  # Write the encrypted key string, conformant to
  # 'securesystemslib.formats.ENCRYPTEDKEY_SCHEMA', to '<filepath>'.
  file_object = securesystemslib.util.TempFile()
  # Raise 'securesystemslib.exceptions.CryptoError' if 'ecdsa_key' cannot be
  # encrypted.
  encrypted_key = securesystemslib.keys.encrypt_key(ecdsa_key, password)
  file_object.write(encrypted_key.encode('utf-8'))
  file_object.move(filepath)

  return filepath