def close(self, request, *args, **kwargs):
        """
        To close alert - run **POST** against */api/alerts/<alert_uuid>/close/*. No data is required.
        Only users with staff privileges can close alerts.
        """
        if not request.user.is_staff:
            raise PermissionDenied()
        alert = self.get_object()
        alert.close()

        return response.Response(status=status.HTTP_204_NO_CONTENT)