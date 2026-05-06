def base_url(klass, space_id='', resource_id=None, environment_id=None, **kwargs):
        """
        Returns the URI for the resource.
        """

        url = "spaces/{0}".format(
            space_id)

        if environment_id is not None:
            url = url = "{0}/environments/{1}".format(url, environment_id)

        url = "{0}/{1}".format(
            url,
            base_path_for(klass.__name__)
        )

        if resource_id:
            url = "{0}/{1}".format(url, resource_id)

        return url