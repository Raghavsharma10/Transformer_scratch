def list(self, request, *args, **kwargs):
        """
        Filter services by type
        ^^^^^^^^^^^^^^^^^^^^^^^

        It is possible to filter services by their types. Example:

          /api/services/?service_type=DigitalOcean&service_type=OpenStack
        """
        return super(ServicesViewSet, self).list(request, *args, **kwargs)