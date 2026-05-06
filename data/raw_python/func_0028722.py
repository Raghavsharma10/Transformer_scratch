def import_ecdsakey_from_private_pem(pem, scheme='ecdsa-sha2-nistp256', password=None):
  """
  <Purpose>
    Import the private ECDSA key stored in 'pem', and generate its public key
    (which will also be included in the returned ECDSA key object).  In addition,
    a keyid identifier for the ECDSA key is generated.  The object returned
    conforms to:

    {'keytype': 'ecdsa-sha2-nistp256',
     'scheme': 'ecdsa-sha2-nistp256',
     'keyid': keyid,
     'keyval': {'public': '-----BEGIN PUBLIC KEY----- ... -----END PUBLIC KEY-----',
                'private': '-----BEGIN EC PRIVATE KEY----- ... -----END EC PRIVATE KEY-----'}}

    The private key is a string in PEM format.

    >>> ecdsa_key = generate_ecdsa_key()
    >>> private_pem = ecdsa_key['keyval']['private']
    >>> ecdsa_key = import_ecdsakey_from_private_pem(private_pem)
    >>> securesystemslib.formats.ECDSAKEY_SCHEMA.matches(ecdsa_key)
    True

  <Arguments>
    pem:
      A string in PEM format.  The private key is extracted and returned in
      an ecdsakey object.

    scheme:
      The signature scheme used by the imported key.

    password: (optional)
      The password, or passphrase, to decrypt the private part of the ECDSA
      key if it is encrypted.  'password' is not used directly as the encryption
      key, a stronger encryption key is derived from it.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if 'pem' specifies
    an unsupported key type.

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

  # Is 'scheme' properly formatted?
  securesystemslib.formats.ECDSA_SCHEME_SCHEMA.check_match(scheme)

  if password is not None:
    securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  else:
    logger.debug('The password/passphrase is unset.  The PEM is expected'
      ' to be unencrypted.')

  # Begin building the ECDSA key dictionary.
  ecdsakey_dict = {}
  keytype = 'ecdsa-sha2-nistp256'
  public = None
  private = None

  public, private = \
      securesystemslib.ecdsa_keys.create_ecdsa_public_and_private_from_pem(pem,
      password)

  # Generate the keyid of the ECDSA key.  'key_value' corresponds to the
  # 'keyval' entry of the 'ECDSAKEY_SCHEMA' dictionary.  The private key
  # information is not included in the generation of the 'keyid' identifier.
  # Convert any '\r\n' (e.g., Windows) newline characters to '\n' so that a
  # consistent keyid is generated.
  key_value = {'public': public.replace('\r\n', '\n'),
               'private': ''}
  keyid = _get_keyid(keytype, scheme, key_value)

  # Build the 'ecdsakey_dict' dictionary.  Update 'key_value' with the ECDSA
  # private key prior to adding 'key_value' to 'ecdsakey_dict'.
  key_value['private'] = private

  ecdsakey_dict['keytype'] = keytype
  ecdsakey_dict['scheme'] = scheme
  ecdsakey_dict['keyid'] = keyid
  ecdsakey_dict['keyval'] = key_value

  # Add "keyid_hash_algorithms" so equal ECDSA keys with
  # different keyids can be associated using supported keyid_hash_algorithms
  ecdsakey_dict['keyid_hash_algorithms'] = \
    securesystemslib.settings.HASH_ALGORITHMS

  return ecdsakey_dict