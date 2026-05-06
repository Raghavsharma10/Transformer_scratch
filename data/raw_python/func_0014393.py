def base_url(self, space_id, content_type_id, environment_id=None, **kwargs):
        """
        Returns the URI for the editor interface.
        """

        return "spaces/{0}{1}/content_types/{2}/editor_interface".format(
            space_id,
            '/environments/{0}'.format(environment_id) if environment_id is not None else '',
            content_type_id
        )