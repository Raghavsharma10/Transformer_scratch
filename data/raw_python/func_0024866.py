def create_manifest(self):
        """
        Create a new manifest and write it to
        disk.
        """
        self.manifest = {}
        self.manifest['applications'] = [{'name': self.app_name}]
        self.manifest['services'] = []
        self.manifest['env'] = {
                'PREDIXPY_VERSION': str(predix.version),
                }

        self.write_manifest()