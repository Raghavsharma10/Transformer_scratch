def write_manifest(self, manifest_path=None, encrypted=None):
        """
        Write manifest to disk.

        :param manifest_path: write to a different location
        :param encrypted: write with env data encrypted

        """
        manifest_path = manifest_path or self.manifest_path
        self.manifest['env']['PREDIXPY_VERSION'] = str(predix.version)

        with open(manifest_path, 'w') as output_file:
            if encrypted or self.encrypted:
                self.manifest['env']['PREDIXPY_ENCRYPTED'] = self.manifest_key
                content = self._get_encrypted_manifest()
            else:
                content = self.manifest   # shallow reference
                if 'PREDIXPY_ENCRYPTED' in content['env']:
                    del(content['env']['PREDIXPY_ENCRYPTED'])

            yaml.safe_dump(content, output_file,
                    default_flow_style=False, explicit_start=True)
            output_file.close()