def form_valid(self, form):
        """
        Processes a valid form submittal.

        :param form: the form instance.
        :rtype: django.http.HttpResponse.
        """
        #noinspection PyAttributeOutsideInit
        self.object = form.save()

        # Index the link.
        for backend in get_search_backends():
            backend.add(self.object)

        #noinspection PyUnresolvedReferences
        return render_modal_workflow(
            self.request,
            None,
            '{0}/chosen.js'.format(self.template_dir),
            {'obj': self.get_json(self.object)}
        )