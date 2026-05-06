def create_virtualenv(self):
        """
        Populate venv from preloaded image
        """
        return task.create_virtualenv(self.target, self.datadir,
            self._preload_image(), self._get_container_name)