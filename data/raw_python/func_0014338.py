def base_url(klass, space_id, resource_id=None, public=False, environment_id=None, **kwargs):
        """
        Returns the URI for the content type.
        """

        if public:
            environment_slug = ""
            if environment_id is not None:
                environment_slug = "/environments/{0}".format(environment_id)
            return "spaces/{0}{1}/public/content_types".format(space_id, environment_slug)
        return super(ContentType, klass).base_url(
            space_id,
            resource_id=resource_id,
            environment_id=environment_id,
            **kwargs
        )