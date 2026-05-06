def jarsign(storepass, keypass, keystore, source, alias, path=None):
  """
  Uses Jarsign to sign an apk target file using the provided keystore information.

  :param storepass(str) - keystore storepass
  :param keypass(str) - keystore keypass
  :param keystore(str) - keystore file path
  :param source(str) - apk path
  :param alias(str) - keystore alias
  :param path(str) - basedir to run the command
  """
  cmd = [
    'jarsigner',
    '-verbose',
    '-storepass',
    storepass,
    '-keypass',
    keypass,
    '-keystore',
    keystore,
    source,
    alias
  ]
  common.run_cmd(cmd, log='jarsign.log', cwd=path)