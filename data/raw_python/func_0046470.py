def shared_volume_containers(self):
        """All the harpoon containers in volumes.share_with for this container"""
        for container in self.volumes.share_with:
            if not isinstance(container, six.string_types):
                yield container.name