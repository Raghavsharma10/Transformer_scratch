def base_url(klass, space_id, parent_resource_id, resource_url='entries', resource_id=None, environment_id=None):
        """
        Returns the URI for the snapshot.
        """

        return "spaces/{0}{1}/{2}/{3}/snapshots/{4}".format(
            space_id,
            '/environments/{0}'.format(environment_id) if environment_id is not None else '',
            resource_url,
            parent_resource_id,
            resource_id if resource_id is not None else ''
        )