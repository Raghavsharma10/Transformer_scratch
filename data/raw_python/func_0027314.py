def acknowledge(self, request, *args, **kwargs):
        """
        To acknowledge alert - run **POST** against */api/alerts/<alert_uuid>/acknowledge/*. No payload is required.
        All users that can see alerts can also acknowledge it. If alert is already acknowledged endpoint
        will return error with code 409(conflict).
        """
        alert = self.get_object()
        if not alert.acknowledged:
            alert.acknowledge()
            return response.Response(status=status.HTTP_200_OK)
        else:
            return response.Response({'detail': _('Alert is already acknowledged.')}, status=status.HTTP_409_CONFLICT)