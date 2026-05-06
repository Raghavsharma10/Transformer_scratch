def get_object_by_filename(self, request, filename):
        """
        Returns owner object by filename (to be downloaded).
        This can be used to implement custom permission checks.
        :param request: HttpRequest
        :param filename: File name of the downloaded object.
        :return: owner object
        """
        kw = dict()
        kw[self.file_field] = filename
        obj = self.get_queryset(request).filter(**kw).first()
        if not obj:
            raise Http404(_('File {} not found').format(filename))
        return self.get_object(request, obj.id)