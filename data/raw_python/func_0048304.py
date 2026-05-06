def alter_1(self, given_container_name, container_name, meta, val):
        """Get the container_name of the container if a container is specified"""
        meta.container = None
        if not isinstance(container_name, six.string_types):
            meta.container = container_name
            container_name = container_name.container_name
        return container_name