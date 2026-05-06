def add_extra_container(self, container, error_on_exists=False):
        """
        Add a container as a 'extra'. These are running containers which are not necessary for
        running default CKAN but are useful for certain extensions
        :param container: The container name to add
        :param error_on_exists: Raise a DatacatsError if the extra container already exists.
        """
        if container in self.extra_containers:
            if error_on_exists:
                raise DatacatsError('{} is already added as an extra container.'.format(container))
            else:
                return

        self.extra_containers.append(container)

        cp = SafeConfigParser()
        cp.read(self.target + '/.datacats-environment')

        cp.set('datacats', 'extra_containers', ' '.join(self.extra_containers))

        with open(self.target + '/.datacats-environment', 'w') as f:
            cp.write(f)