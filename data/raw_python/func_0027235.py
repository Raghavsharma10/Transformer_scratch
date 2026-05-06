def unlink(self, request, uuid=None):
        """
        Unlink all related resources, service project link and service itself.
        """
        service = self.get_object()
        service.unlink_descendants()
        self.perform_destroy(service)

        return Response(status=status.HTTP_204_NO_CONTENT)