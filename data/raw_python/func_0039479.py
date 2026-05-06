def get_default_keystore(prefix='AG_'):
  """
  Gets the default keystore information based on environment variables and a prefix.

  $PREFIX_KEYSTORE_PATH - keystore file path, default is opt/digger/debug.keystore
  $PREFIX_KEYSTORE_STOREPASS - keystore storepass, default is android
  $PREFIX_KEYSTORE_KEYPASS - keystore keypass, default is android
  $PREFIX_KEYSTORE_ALIAS - keystore alias, default is androiddebug
  
  :param prefix(str) - A prefix to be used for environment variables, default is AG_.

  Returns:
    A tuple containing the keystore information: (path, storepass, keypass, alias)
  """
  path = os.environ.get('%s_KEYSTORE_PATH' % prefix, config.keystore.path)
  storepass = os.environ.get('%s_KEYSTORE_STOREPASS' % prefix, config.keystore.storepass)
  keypass = os.environ.get('%s_KEYSTORE_KEYPASS' % prefix, config.keystore.keypass)
  alias = os.environ.get('%s_KEYSTORE_ALIAS' % prefix, config.keystore.alias)
  return (path, storepass, keypass, alias)