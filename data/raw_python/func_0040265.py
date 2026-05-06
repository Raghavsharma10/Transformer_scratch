def get_success_headers(self, data):
        """
        As per the *Annotator* documentation regarding the
        `create <http://docs.annotatorjs.org/en/v1.2.x/storage.html#create>`_
        and
        `update <http://docs.annotatorjs.org/en/v1.2.x/storage.html#update>`_
        endpoints, we must return an absolute URL in the ``Location``
        header.
        :param data:
            serialized object.
        :return:
            :class:`dict` of HTTP headers.
        """
        headers = super(AnnotationViewSet, self).get_success_headers(data)

        url = urlresolvers.reverse("annotations-detail",
                                   kwargs={"pk": data["id"]})
        headers.update({"Location": self.request.build_absolute_uri(url)})

        return headers