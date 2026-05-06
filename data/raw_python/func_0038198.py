def trash(self, request, **kwargs):
        """Psuedo-deletes a `Content` instance and removes it from the ElasticSearch index

        Content is not actually deleted, merely hidden by deleted from ES index.import

        :param request: a WSGI request object
        :param kwargs: keyword arguments (optional)
        :return: `rest_framework.response.Response`
        """
        content = self.get_object()

        content.indexed = False
        content.save()

        LogEntry.objects.log(request.user, content, "Trashed")
        return Response({"status": "Trashed"})