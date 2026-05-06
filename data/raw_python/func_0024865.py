def read_manifest(self, encrypted=None):
        """
        Read an existing manifest.
        """
        with open(self.manifest_path, 'r') as input_file:
            self.manifest = yaml.safe_load(input_file)
            if 'env' not in self.manifest:
                self.manifest['env'] = {}
            if 'services' not in self.manifest:
                self.manifest['services'] = []

            # If manifest is encrypted, use manifest key to
            # decrypt each value before storing in memory.

            if 'PREDIXPY_ENCRYPTED' in self.manifest['env']:
                self.encrypted = True

            if encrypted or self.encrypted:
                key = predix.config.get_crypt_key(self.manifest_key)
                f = Fernet(key)

                for var in self.manifest['env'].keys():
                    value = f.decrypt(bytes(self.manifest['env'][var], 'utf-8'))
                    self.manifest['env'][var] = value.decode('utf-8')

            self.app_name = self.manifest['applications'][0]['name']

            input_file.close()