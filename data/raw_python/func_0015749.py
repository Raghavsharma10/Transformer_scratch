def form_invalid(self, form):
        """
        Processes an invalid form submittal.

        :param form: the form instance.
        :rtype: django.http.HttpResponse.
        """
        meta = getattr(self.model, '_meta')

        #noinspection PyUnresolvedReferences
        messages.error(
            self.request,
            _(u'The {0} could not be saved due to errors.').format(
                meta.verbose_name.lower()
            )
        )

        return super(BaseEditView, self).form_invalid(form)