def form_valid(self, form):
        """
        Processes a valid form submittal.

        :param form: the form instance.
        :rtype: django.http.HttpResponse.
        """
        #noinspection PyAttributeOutsideInit
        self.object = form.save()
        meta        = getattr(self.object, '_meta')

        # Index the object.
        for backend in get_search_backends():
            backend.add(object)

        #noinspection PyUnresolvedReferences
        messages.success(
            self.request,
            _(u'{0} "{1}" saved.').format(
                meta.verbose_name,
                str(self.object)
            ),
            buttons=[messages.button(
                reverse(
                    '{0}:edit'.format(self.url_namespace),
                    args=(self.object.id,)
                ),
                _(u'Edit')
            )]
        )

        return redirect(self.get_success_url())