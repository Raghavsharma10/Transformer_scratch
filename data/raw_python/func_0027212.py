def retrieve(self, request, *args, **kwargs):
        """
        Optional `field` query parameter (can be list) allows to limit what fields are returned.
        For example, given request /api/projects/<uuid>/?field=uuid&field=name you get response like this:

        .. code-block:: javascript

            {
                "uuid": "90bcfe38b0124c9bbdadd617b5d739f5",
                "name": "Default"
            }
        """
        return super(ProjectViewSet, self).retrieve(request, *args, **kwargs)