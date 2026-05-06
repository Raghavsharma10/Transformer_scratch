def format_keyval_to_metadata(keytype, scheme, key_value, private=False):
  """
  <Purpose>
    Return a dictionary conformant to 'securesystemslib.formats.KEY_SCHEMA'.
    If 'private' is True, include the private key.  The dictionary
    returned has the form:

    {'keytype': keytype,
     'scheme' : scheme,
     'keyval': {'public': '...',
                'private': '...'}}

    or if 'private' is False:

    {'keytype': keytype,
     'scheme': scheme,
     'keyval': {'public': '...',
                'private': ''}}

    >>> ed25519_key = generate_ed25519_key()
    >>> key_val = ed25519_key['keyval']
    >>> keytype = ed25519_key['keytype']
    >>> scheme = ed25519_key['scheme']
    >>> ed25519_metadata = \
    format_keyval_to_metadata(keytype, scheme, key_val, private=True)
    >>> securesystemslib.formats.KEY_SCHEMA.matches(ed25519_metadata)
    True

  <Arguments>
    key_type:
      The 'rsa' or 'ed25519' strings.

    scheme:
      The signature scheme used by the key.

    key_value:
      A dictionary containing a private and public keys.
      'key_value' is of the form:

      {'public': '...',
       'private': '...'}},

      conformant to 'securesystemslib.formats.KEYVAL_SCHEMA'.

    private:
      Indicates if the private key should be included in the dictionary
      returned.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'key_value' does not conform to
    'securesystemslib.formats.KEYVAL_SCHEMA', or if the private key is not
    present in 'key_value' if requested by the caller via 'private'.

  <Side Effects>
    None.

  <Returns>
    A 'securesystemslib.formats.KEY_SCHEMA' dictionary.
  """

  # Does 'keytype' have the correct format?
  # This check will ensure 'keytype' has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.KEYTYPE_SCHEMA.check_match(keytype)

  # Does 'scheme' have the correct format?
  securesystemslib.formats.SCHEME_SCHEMA.check_match(scheme)

  # Does 'key_value' have the correct format?
  securesystemslib.formats.KEYVAL_SCHEMA.check_match(key_value)

  if private is True:
    # If the caller requests (via the 'private' argument) to include a private
    # key in the returned dictionary, ensure the private key is actually
    # present in 'key_val' (a private key is optional for 'KEYVAL_SCHEMA'
    # dicts).
    if 'private' not in key_value:
      raise securesystemslib.exceptions.FormatError('The required private key'
        ' is missing from: ' + repr(key_value))

    else:
      return {'keytype': keytype, 'scheme': scheme, 'keyval': key_value}

  else:
    public_key_value = {'public': key_value['public']}

    return {'keytype': keytype,
            'scheme': scheme,
            'keyid_hash_algorithms': securesystemslib.settings.HASH_ALGORITHMS,
            'keyval': public_key_value}