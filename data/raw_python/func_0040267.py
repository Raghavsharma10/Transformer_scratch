def update(self, request, *args, **kwargs):
        """
        See the *Annotator* documentation regarding the
        `update <http://docs.annotatorjs.org/en/v1.2.x/storage.html#update>`_
        endpoint.

        :param request:
            incoming :class:`rest_framework.request.Request`.
        :return:
            303 :class:`rest_framework.response.Response`.
        """
        response = super(AnnotationViewSet, self).update(request,
                                                         *args,
                                                         **kwargs)
        for h, v in self.get_success_headers(response.data).items():
            response[h] = v
        response.data = None
        response.status_code = status.HTTP_303_SEE_OTHER
        return response