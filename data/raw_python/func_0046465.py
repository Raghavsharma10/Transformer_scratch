def container_id(self):
        """
        Find a container id

        If one isn't already set, we ask docker for the container whose name is
        the same as the recorded container_name
        """
        if getattr(self, "_container_id", None):
            return self._container_id

        try:
            containers = self.harpoon.docker_api.containers(all=True)
        except ValueError:
            log.warning("Failed to get a list of active docker files")
            containers = []

        self._container_id = None
        for container in containers:
            if any(self.name in container.get("Names", []) for name in (self.container_name, "/{0}".format(self.container_name))):
                self._container_id = container["Id"]
                break

        return self._container_id