def delete(self, request, *args, **kwargs):
        """
        Processes deletion of the specified instance.

        :param request: the request instance.
        :rtype: django.http.HttpResponse.
        """
        #noinspection PyAttributeOutsideInit
        self.object = self.get_object()
        success_url = self.get_success_url()
        meta        = getattr(self.object, '_meta')

        self.object.delete()

        messages.success(
            request,
            _(u'{0} "{1}" deleted.').format(
                meta.verbose_name.lower(),
                str(self.object)
            )
        )

        return redirect(success_url)