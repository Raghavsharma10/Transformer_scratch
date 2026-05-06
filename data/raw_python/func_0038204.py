def create(self, request, *args, **kwargs):
        """
        Grabbing the user from request.user, and the rest of the method
        is the same as ModelViewSet.create().

        :param request: a WSGI request object
        :param args: inline arguments (optional)
        :param kwargs: keyword arguments (optional)
        :return: `rest_framework.response.Response`
        :raise: 400
        """
        data = get_request_data(request).copy()
        data["user"] = request.user.id
        serializer = self.get_serializer(data=data, files=request.FILES)

        if serializer.is_valid():
            self.pre_save(serializer.object)
            self.object = serializer.save(force_insert=True)
            self.post_save(self.object, created=True)
            headers = self.get_success_headers(serializer.data)
            return Response(
                serializer.data,
                status=status.HTTP_201_CREATED,
                headers=headers
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)