def get_associations(self, env):
        """
        Get all the associations for this env.

        Root cannot have associations, so return None for root.

        returns a map of hostnames to environments.
        """

        if env.is_root:
            return None

        associations = self.associations.get_all()
        return [assoc for assoc in associations
                if associations[assoc] == self._get_view_path(env)]