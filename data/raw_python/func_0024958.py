def lock_to_org_space(self):
        """
        Lock the manifest to the current organization and space regardless of
        Cloud Foundry target.
        """
        self.add_env_var('PREDIX_ORGANIZATION_GUID', self.space.org.guid)
        self.add_env_var('PREDIX_ORGANIZATION_NAME', self.space.org.name)
        self.add_env_var('PREDIX_SPACE_GUID', self.space.guid)
        self.add_env_var('PREDIX_SPACE_NAME', self.space.name)
        self.write_manifest()