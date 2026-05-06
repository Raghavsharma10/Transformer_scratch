def connect_to_rackspace(region,
                         access_key_id,
                         secret_access_key):
    """ returns a connection object to Rackspace  """
    pyrax.set_setting('identity_type', 'rackspace')
    pyrax.set_default_region(region)
    pyrax.set_credentials(access_key_id, secret_access_key)
    nova = pyrax.connect_to_cloudservers(region=region)
    return nova