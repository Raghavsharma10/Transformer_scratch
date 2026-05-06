def validate_env(self, envname):
        """
        Check the name of the environment against the black list and the
        whitelist. If a whitelist is specified only it is checked.
        """
        if self.whitelist_envs and envname in self.whitelist_envs:
            return True
        elif self.whitelist_envs:
            return False

        if self.blacklist_envs and envname not in self.blacklist_envs:
            return True
        elif self.blacklist_envs:
            # If there is just a True, all envs are blacklisted
            return False
        else:
            return True