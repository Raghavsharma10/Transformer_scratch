def body(cls, description=None, default=None, resource=DefaultResource, **options):
        """
        Define body parameter.
        """
        return cls('body', In.Body, None, resource, description, required=True,
                   default=default, **options)