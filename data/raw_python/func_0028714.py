def import_rsakey_from_private_pem(pem, scheme='rsassa-pss-sha256', password=None):
  """
  <Purpose>
    Import the private RSA key stored in 'pem', and generate its public key
    (which will also be included in the returned rsakey object).  In addition,
    a keyid identifier for the RSA key is generated.  The object returned
    conforms to 'securesystemslib.formats.RSAKEY_SCHEMA' and has the form:

    {'keytype': 'rsa',
     'scheme': 'rsassa-pss-sha256',
     'keyid': keyid,
     'keyval': {'public': '-----BEGIN RSA PUBLIC KEY----- ...',
                'private': '-----BEGIN RSA PRIVATE KEY----- ...'}}

    The private key is a string in PEM format.

    >>> rsa_key = generate_rsa_key()
    >>> scheme = rsa_key['scheme']
    >>> private = rsa_key['keyval']['private']
    >>> passphrase = 'secret'
    >>> encrypted_pem = create_rsa_encrypted_pem(private, passphrase)
    >>> rsa_key2 = import_rsakey_from_private_pem(encrypted_pem, scheme, passphrase)
    >>> securesystemslib.formats.RSAKEY_SCHEMA.matches(rsa_key)
    True
    >>> securesystemslib.formats.RSAKEY_SCHEMA.matches(rsa_key2)
    True

  <Arguments>
    pem:
      A string in PEM format.  The private key is extracted and returned in
      an rsakey object.

    scheme:
      The signature scheme used by the imported key.

    password: (optional)
      The password, or passphrase, to decrypt the private part of the RSA key
      if it is encrypted.  'password' is not used directly as the encryption
      key, a stronger encryption key is derived from it.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the arguments are improperly
    formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if 'pem' specifies
    an unsupported key type.

  <Side Effects>
    None.

  <Returns>
    A dictionary containing the RSA keys and other identifying information.
    Conforms to 'securesystemslib.formats.RSAKEY_SCHEMA'.
  """

  # Does 'pem' have the correct format?
  # This check will ensure 'pem' conforms to
  # 'securesystemslib.formats.PEMRSA_SCHEMA'.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(pem)

  # Is 'scheme' properly formatted?
  securesystemslib.formats.RSA_SCHEME_SCHEMA.check_match(scheme)

  if password is not None:
    securesystemslib.formats.PASSWORD_SCHEMA.check_match(password)

  else:
    logger.debug('The password/passphrase is unset.  The PEM is expected'
      ' to be unencrypted.')

  # Begin building the RSA key dictionary.
  rsakey_dict = {}
  keytype = 'rsa'
  public = None
  private = None

  # Generate the public and private RSA keys.  The pyca/cryptography library
  # performs the actual crypto operations.
  public, private = \
    securesystemslib.pyca_crypto_keys.create_rsa_public_and_private_from_pem(
    pem, password)

  public =  extract_pem(public, private_pem=False)
  private = extract_pem(private, private_pem=True)

  # Generate the keyid of the RSA key.  'key_value' corresponds to the
  # 'keyval' entry of the 'RSAKEY_SCHEMA' dictionary.  The private key
  # information is not included in the generation of the 'keyid' identifier.
  # Convert any '\r\n' (e.g., Windows) newline characters to '\n' so that a
  # consistent keyid is generated.
  key_value = {'public': public.replace('\r\n', '\n'),
               'private': ''}
  keyid = _get_keyid(keytype, scheme, key_value)

  # Build the 'rsakey_dict' dictionary.  Update 'key_value' with the RSA
  # private key prior to adding 'key_value' to 'rsakey_dict'.
  key_value['private'] = private

  rsakey_dict['keytype'] = keytype
  rsakey_dict['scheme'] = scheme
  rsakey_dict['keyid'] = keyid
  rsakey_dict['keyval'] = key_value

  return rsakey_dict