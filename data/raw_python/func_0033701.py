def _set_WorkingDir(self, path):
        """Sets the working directory
        """
        self._curr_working_dir = path
        try:
            mkdir(self.WorkingDir)
        except OSError:
            # Directory already exists
            pass