def count(self):
        """Analyze the counts of ...things.

        Returns
        -------
        retvals : dict
            Dictionary of 'property-name: counts' pairs for further processing

        """
        self.log.info("Running 'count'")
        retvals = {}

        # Numbers of 'tasks'
        num_tasks = self._count_tasks()
        retvals['num_tasks'] = num_tasks

        # Numbers of 'files'
        num_files = self._count_repo_files()
        retvals['num_files'] = num_files

        return retvals