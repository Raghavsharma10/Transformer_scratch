def create_token(self, request, **kwargs):
        """Create a new obfuscated url info to use for accessing unpublished content.

        :param request: a WSGI request object
        :param kwargs: keyword arguments (optional)
        :return: `rest_framework.response.Response`
        """

        data = {
            "content": self.get_object().id,
            "create_date": get_request_data(request)["create_date"],
            "expire_date": get_request_data(request)["expire_date"]
        }
        serializer = ObfuscatedUrlInfoSerializer(data=data)
        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST,
                content_type="application/json",
            )
        serializer.save()

        return Response(serializer.data, status=status.HTTP_200_OK, content_type="application/json")