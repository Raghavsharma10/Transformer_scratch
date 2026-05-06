def status(self, request, **kwargs):
        """This endpoint returns a status text, currently one of:
          - "Draft" (If no publish date is set, and no item exists in the editor queue)
          - "Waiting for Editor" (If no publish date is set, and an item exists in the editor queue)
          - "Published" (The published date is in the past)
          - "Scheduled" (The published date is set in the future)

        :param request: a WSGI request object
        :param kwargs: keyword arguments (optional)
        :return: `rest_framework.response.Response`
        """
        content = self.get_object()
        return Response({"status": content.get_status()})