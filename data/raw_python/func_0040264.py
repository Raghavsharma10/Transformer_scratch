def search(self, _):
        """
        Implements the
        `search <http://docs.annotatorjs.org/en/v1.2.x/storage.html#search>`_
        endpoint.

        We rely on the behaviour of the ``filter_backends`` to manage
        the actual filtering of search results.

        :param _:
            :class:`rest_framework.request.Request` object—ignored here
            as we rely on the ``filter_backends``.
        :return:
            filtered :class:`rest_framework.response.Response`.
        """
        queryset = super(AnnotationViewSet, self).filter_queryset(
            self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)

        return Response({
            "total": len(serializer.data),
            "rows": serializer.data
        })