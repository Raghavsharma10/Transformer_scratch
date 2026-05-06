def create(self, request, *args, **kwargs):
        """ We ensure the Thread only involves eligible participants. """
        serializer = self.get_serializer(data=compat_get_request_data(request))
        compat_serializer_check_is_valid(serializer)
        self.perform_create(request, serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)