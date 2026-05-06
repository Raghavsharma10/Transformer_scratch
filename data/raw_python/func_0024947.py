def add_to_manifest(self, manifest):
        """
        Add to the manifest to make sure it is bound to the
        application.
        """
        manifest.add_service(self.service.name)
        manifest.write_manifest()