def _get_default_dependencies(self):
        '''
        Get default dependencies for archive

        Get default dependencies from requirements file or (if no requirements
        file) from previous version
        '''

        # Get default dependencies from requirements file
        default_dependencies = {
            k: v for k,
            v in self.api.default_versions.items() if k != self.archive_name}

        # If no requirements file or is empty:
        if len(default_dependencies) == 0:

            # Retrieve dependencies from last archive record
            history = self.get_history()

            if len(history) > 0:
                default_dependencies = history[-1].get('dependencies', {})

        return default_dependencies