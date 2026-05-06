def contributions(self, request, **kwargs):
        """gets or adds contributions

        :param request: a WSGI request object
        :param kwargs: keyword arguments (optional)
        :return: `rest_framework.response.Response`
        """
        # Check if the contribution app is installed
        if Contribution not in get_models():
            return Response([])

        if request.method == "POST":
            serializer = ContributionSerializer(data=get_request_data(request), many=True)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            serializer.save()
            return Response(serializer.data)
        else:
            content_pk = kwargs.get('pk', None)
            if content_pk is None:
                return Response([], status=status.HTTP_404_NOT_FOUND)
            queryset = Contribution.search_objects.search().filter(
                es_filter.Term(**{'content.id': content_pk})
            )
            serializer = ContributionSerializer(queryset[:queryset.count()].sort('id'), many=True)
            return Response(serializer.data)