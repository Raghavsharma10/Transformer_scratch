def chosen_view_factory(chooser_cls):
    """
    Returns a ChosenView class that extends specified chooser class.

    :param chooser_cls: the class to extend.
    :rtype: class.
    """
    class ChosenView(chooser_cls):
        #noinspection PyUnusedLocal
        def get(self, request, *args, **kwargs):
            """
            Returns GET response.

            :param request: the request instance.
            :rtype: django.http.HttpResponse.
            """
            #noinspection PyAttributeOutsideInit
            self.object = self.get_object()

            return render_modal_workflow(
                self.request,
                None,
                '{0}/chosen.js'.format(self.template_dir),
                {'obj': self.get_json(self.object)}
            )

        def get_object(self, queryset=None):
            """
            Returns chosen object instance.

            :param queryset: the queryset instance.
            :rtype: django.db.models.Model.
            """
            if queryset is None:
                queryset = self.get_queryset()

            pk = self.kwargs.get('pk', None)

            try:
                return queryset.get(pk=pk)
            except self.models.DoesNotExist:
                raise Http404()

        def post(self, request, *args, **kwargs):
            """
            Returns POST response.

            :param request: the request instance.
            :rtype: django.http.HttpResponse.
            """
            return self.get(request, *args, **kwargs)

    return ChosenView