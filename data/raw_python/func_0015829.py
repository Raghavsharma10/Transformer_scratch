def post(self, request, *args, **kwargs):
        """
        Returns POST response.

        :param request: the request instance.
        :rtype: django.http.HttpResponse.
        """
        form        = None
        link_type   = int(request.POST.get('link_type', 0))

        if link_type == Link.LINK_TYPE_EMAIL:
            form = EmailLinkForm(**self.get_form_kwargs())
        elif link_type == Link.LINK_TYPE_EXTERNAL:
            form = ExternalLinkForm(**self.get_form_kwargs())

        if form:
            if form.is_valid():
                return self.form_valid(form)
            else:
                return self.form_invalid(form)
        else:
            raise Http404()