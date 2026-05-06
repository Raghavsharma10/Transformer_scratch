def _get_encrypted_manifest(self):
        """
        Returns contents of the manifest where environment variables
        that are secret will be encrypted without modifying the existing
        state in memory which will remain unencrypted.
        """
        key = predix.config.get_crypt_key(self.manifest_key)
        f = Fernet(key)

        manifest = copy.deepcopy(self.manifest)
        for var in self.manifest['env'].keys():
            value = str(self.manifest['env'][var])
            manifest['env'][var] = f.encrypt(bytes(value, 'utf-8')).decode('utf-8')

        return manifest