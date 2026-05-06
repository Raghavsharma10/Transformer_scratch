def sign(self, storepass=None, keypass=None, keystore=None, apk=None, alias=None, name='app'):
    """
    Signs (jarsign and zipalign) a target apk file based on keystore information, uses default debug keystore file by default.

    :param storepass(str): keystore file storepass
    :param keypass(str): keystore file keypass
    :param keystore(str): keystore file path
    :param apk(str): apk file path to be signed
    :param alias(str): keystore file alias
    :param name(str): signed apk name to be used by zipalign
    """
    target = self.get_target()
    build_tool = android_helper.get_highest_build_tool(target.split('-')[1])
    if keystore is None:
      (keystore, storepass, keypass, alias) = android_helper.get_default_keystore()
    dist = '%s/%s.apk' % ('/'.join(apk.split('/')[:-1]), name)
    android_helper.jarsign(storepass, keypass, keystore, apk, alias, path=self.path)
    android_helper.zipalign(apk, dist, build_tool=build_tool, path=self.path)