def create_directories(self, create_project_dir=True):
        """
        Call once for new projects to create the initial project directories.
        """
        return task.create_directories(self.datadir, self.sitedir,
            self.target if create_project_dir else None)