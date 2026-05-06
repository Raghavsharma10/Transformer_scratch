def _unpack_user(v):
  """Internal helper to unpack a User value from a protocol buffer."""
  uv = v.uservalue()
  email = unicode(uv.email().decode('utf-8'))
  auth_domain = unicode(uv.auth_domain().decode('utf-8'))
  obfuscated_gaiaid = uv.obfuscated_gaiaid().decode('utf-8')
  obfuscated_gaiaid = unicode(obfuscated_gaiaid)

  federated_identity = None
  if uv.has_federated_identity():
    federated_identity = unicode(
        uv.federated_identity().decode('utf-8'))

  value = users.User(email=email,
                     _auth_domain=auth_domain,
                     _user_id=obfuscated_gaiaid,
                     federated_identity=federated_identity)
  return value