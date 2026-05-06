def format_metadata_to_key(key_metadata):
  """
  <Purpose>
    Construct a key dictionary (e.g., securesystemslib.formats.RSAKEY_SCHEMA)
    according to the keytype of 'key_metadata'.  The dict returned by this
    function has the exact format as the dict returned by one of the key
    generations functions, like generate_ed25519_key().  The dict returned
    has the form:

    {'keytype': keytype,
     'scheme': scheme,
     'keyid': 'f30a0870d026980100c0573bd557394f8c1bbd6...',
     'keyval': {'public': '...',
                'private': '...'}}

    For example, RSA key dictionaries in RSAKEY_SCHEMA format should be used by
    modules storing a collection of keys, such as with keydb.py.  RSA keys as
    stored in metadata files use a different format, so this function should be
    called if an RSA key is extracted from one of these metadata files and need
    converting.  The key generation functions create an entirely new key and
    return it in the format appropriate for 'keydb.py'.

    >>> ed25519_key = generate_ed25519_key()
    >>> key_val = ed25519_key['keyval']
    >>> keytype = ed25519_key['keytype']
    >>> scheme = ed25519_key['scheme']
    >>> ed25519_metadata = \
    format_keyval_to_metadata(keytype, scheme, key_val, private=True)
    >>> ed25519_key_2, junk = format_metadata_to_key(ed25519_metadata)
    >>> securesystemslib.formats.ED25519KEY_SCHEMA.matches(ed25519_key_2)
    True
    >>> ed25519_key == ed25519_key_2
    True

  <Arguments>
    key_metadata:
      The key dictionary as stored in Metadata files, conforming to
      'securesystemslib.formats.KEY_SCHEMA'.  It has the form:

      {'keytype': '...',
       'scheme': scheme,
       'keyval': {'public': '...',
                  'private': '...'}}

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'key_metadata' does not conform
    to 'securesystemslib.formats.KEY_SCHEMA'.

  <Side Effects>
    None.

  <Returns>
    In the case of an RSA key, a dictionary conformant to
    'securesystemslib.formats.RSAKEY_SCHEMA'.
  """

  # Does 'key_metadata' have the correct format?
  # This check will ensure 'key_metadata' has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.KEY_SCHEMA.check_match(key_metadata)

  # Construct the dictionary to be returned.
  key_dict = {}
  keytype = key_metadata['keytype']
  scheme = key_metadata['scheme']
  key_value = key_metadata['keyval']

  # Convert 'key_value' to 'securesystemslib.formats.KEY_SCHEMA' and generate
  # its hash The hash is in hexdigest form.
  default_keyid = _get_keyid(keytype, scheme, key_value)
  keyids = set()
  keyids.add(default_keyid)

  for hash_algorithm in securesystemslib.settings.HASH_ALGORITHMS:
    keyid = _get_keyid(keytype, scheme, key_value, hash_algorithm)
    keyids.add(keyid)

  # All the required key values gathered.  Build 'key_dict'.
  # 'keyid_hash_algorithms'
  key_dict['keytype'] = keytype
  key_dict['scheme'] = scheme
  key_dict['keyid'] = default_keyid
  key_dict['keyid_hash_algorithms'] = securesystemslib.settings.HASH_ALGORITHMS
  key_dict['keyval'] = key_value

  return key_dict, keyids