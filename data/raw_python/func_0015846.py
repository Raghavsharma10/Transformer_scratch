def form_invalid(self, form):
        """
        Processes an invalid form submittal.

        :param form: the form instance.
        :rtype: django.http.HttpResponse.
        """
        context = self.get_context_data(form=form)

        #noinspection PyUnresolvedReferences
        return render_modal_workflow(
            self.request,
            '{0}/chooser.html'.format(self.template_dir),
            '{0}/chooser.js'.format(self.template_dir),
            context
        )