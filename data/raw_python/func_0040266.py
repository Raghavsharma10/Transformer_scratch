def create(self, request, *args, **kwargs):
        """
        See the *Annotator* documentation regarding the
        `create <http://docs.annotatorjs.org/en/v1.2.x/storage.html#create>`_
        endpoint.

        :param request:
            incoming :class:`rest_framework.request.Request`.
        :return:
            303 :class:`rest_framework.response.Response`.
        """
        response = super(AnnotationViewSet, self).create(request,
                                                         *args,
                                                         **kwargs)
        response.data = None
        response.status_code = status.HTTP_303_SEE_OTHER
        return response