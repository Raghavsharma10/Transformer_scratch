def get(self, request, *args, **kwargs):
        """
        Returns GET response.

        :param request: the request instance.
        :rtype: django.http.HttpResponse.
        """
        #noinspection PyAttributeOutsideInit
        self.object_list    = self.get_queryset()
        context             = self.get_context_data(force_search=True)

        if self.form_class:
            context.update({'form': self.get_form()})

        if 'q' in request.GET or 'p' in request.GET:
            return render(
                request,
                '{0}/results.html'.format(self.template_dir),
                context
            )
        else:
            return render_modal_workflow(
                request,
                '{0}/chooser.html'.format(self.template_dir),
                '{0}/chooser.js'.format(self.template_dir),
                context
            )