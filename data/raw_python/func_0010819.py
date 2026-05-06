def get_context_data(self, **kwargs):
        """
        We supplement the normal context data by adding our fields and labels.
        """
        context = super(SmartView, self).get_context_data(**kwargs)

        # derive our field config
        self.field_config = self.derive_field_config()

        # add our fields
        self.fields = self.derive_fields()

        # build up our current parameter string, EXCLUSIVE of our page.  These
        # are used to build pagination URLs
        url_params = "?"
        order_params = ""
        for key in self.request.GET.keys():
            if key != 'page' and key != 'pjax' and (len(key) == 0 or key[0] != '_'):
                for value in self.request.GET.getlist(key):
                    url_params += "%s=%s&" % (key, urlquote(value))
            elif key == '_order':
                order_params = "&".join(["%s=%s" % (key, _) for _ in self.request.GET.getlist(key)])

        context['url_params'] = url_params
        context['order_params'] = order_params + "&"
        context['pjax'] = self.pjax

        # set our blocks
        context['blocks'] = dict()

        # stuff it all in our context
        context['fields'] = self.fields
        context['view'] = self
        context['field_config'] = self.field_config

        context['title'] = self.derive_title()

        # and any extra context the user specified
        context.update(self.extra_context)

        # by default, our base is 'base.html', but we might be pjax
        base_template = "base.html"
        if 'pjax' in self.request.GET or 'pjax' in self.request.POST:
            base_template = "smartmin/pjax.html"

        if 'HTTP_X_PJAX' in self.request.META:
            base_template = "smartmin/pjax.html"

        context['base_template'] = base_template

        # set our refresh if we have one
        refresh = self.derive_refresh()
        if refresh:
            context['refresh'] = refresh

        return context