def save(self):
        """
        Save environment settings into environment directory, overwriting
        any existing configuration and discarding site config
        """
        task.save_new_environment(self.name, self.datadir, self.target,
            self.ckan_version, self.deploy_target, self.always_prod)