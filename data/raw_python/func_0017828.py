def _get_container_name(self, container_type):
        """
        Gets the full name of a container of the type specified.
        Currently the supported types are:
            - 'venv'
            - 'postgres'
            - 'solr'
            - 'web'
            - 'pgdata'
            - 'lessc'
            - 'datapusher'
            - 'redis'
        The name will be formatted appropriately with any prefixes and postfixes
        needed.

        :param container_type: The type of container name to generate (see above).
        """
        if container_type in ['venv']:
            return 'datacats_{}_{}'.format(container_type, self.name)
        else:
            return 'datacats_{}_{}_{}'.format(container_type, self.name, self.site_name)