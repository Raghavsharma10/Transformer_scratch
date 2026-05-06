def generate_ecdsa_key(scheme='ecdsa-sha2-nistp256'):
  """
  <Purpose>
    Generate public and private ECDSA keys, with NIST P-256 + SHA256 (for
    hashing) being the default scheme.  In addition, a keyid identifier for the
    ECDSA key is generated.  The object returned conforms to
    'securesystemslib.formats.ECDSAKEY_SCHEMA' and has the form:

    {'keytype': 'ecdsa-sha2-nistp256',
     'scheme', 'ecdsa-sha2-nistp256',
     'keyid': keyid,
     'keyval': {'public': '',
                'private': ''}}

    The public and private keys are strings in TODO format.

    >>> ecdsa_key = generate_ecdsa_key(scheme='ecdsa-sha2-nistp256')
    >>> securesystemslib.formats.ECDSAKEY_SCHEMA.matches(ecdsa_key)
    True

  <Arguments>
    scheme:
      The ECDSA signature scheme.  By default, ECDSA NIST P-256 is used, with
      SHA256 for hashing.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'scheme' is improperly
    formatted or invalid (i.e., not one of the supported ECDSA signature
    schemes).

  <Side Effects>
    None.

  <Returns>
    A dictionary containing the ECDSA keys and other identifying information.
    Conforms to 'securesystemslib.formats.ECDSAKEY_SCHEMA'.
  """

  # Does 'scheme' have the correct format?
  # This check will ensure 'scheme' is properly formatted and is a supported
  # ECDSA signature scheme.  Raise 'securesystemslib.exceptions.FormatError' if
  # the check fails.
  securesystemslib.formats.ECDSA_SCHEME_SCHEMA.check_match(scheme)

  # Begin building the ECDSA key dictionary.
  ecdsa_key = {}
  keytype = 'ecdsa-sha2-nistp256'
  public = None
  private = None

  # Generate the public and private ECDSA keys with one of the supported
  # libraries.
  public, private = \
    securesystemslib.ecdsa_keys.generate_public_and_private(scheme)

  # Generate the keyid of the Ed25519 key.  'key_value' corresponds to the
  # 'keyval' entry of the 'Ed25519KEY_SCHEMA' dictionary.  The private key
  # information is not included in the generation of the 'keyid' identifier.
  # Convert any '\r\n' (e.g., Windows) newline characters to '\n' so that a
  # consistent keyid is generated.
  key_value = {'public': public.replace('\r\n', '\n'),
               'private': ''}
  keyid = _get_keyid(keytype, scheme, key_value)

  # Build the 'ed25519_key' dictionary.  Update 'key_value' with the Ed25519
  # private key prior to adding 'key_value' to 'ed25519_key'.

  key_value['private'] = private

  ecdsa_key['keytype'] = keytype
  ecdsa_key['scheme'] = scheme
  ecdsa_key['keyid'] = keyid
  ecdsa_key['keyval'] = key_value

  # Add "keyid_hash_algorithms" so that equal ECDSA keys with different keyids
  # can be associated using supported keyid_hash_algorithms.
  ecdsa_key['keyid_hash_algorithms'] = \
      securesystemslib.settings.HASH_ALGORITHMS

  return ecdsa_key