def list_tokens(self, request, **kwargs):
        """List all tokens for this content instance.

        :param request: a WSGI request object
        :param kwargs: keyword arguments (optional)
        :return: `rest_framework.response.Response`
        """

        # no date checking is done here to make it more obvious if there's an issue with the
        # number of records. Date filtering will be done on the frontend.
        infos = [ObfuscatedUrlInfoSerializer(info).data
                 for info in ObfuscatedUrlInfo.objects.filter(content=self.get_object())]
        return Response(infos, status=status.HTTP_200_OK, content_type="application/json")