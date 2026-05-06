def post(self, request, *args, **kwargs):
        """
        Method for handling POST requests.
        If the formset is valid this will
        loop through the formset and save each form.
        A log is generated for each save. The user
        is notified of the total number of changes
        with a message. Returns a 'render redirect' to
        the current url.

        TODO: These formsets suffer from the same potential concurrency
        issues that the django admin has. This is caused by some issues
        with django formsets and concurrent users editing the same
        objects.
        """
        msg = None
        action = request.POST.get('actions', None)
        selected = request.POST.getlist(CHECKBOX_NAME)
        if not action == 'None' and action is not None:
            if len(selected) > 0:
                sel = {CHECKBOX_NAME : ','.join(selected)}
                qs = '?' + urlencode(sel)
                return self.render(request, redirect_url = action + qs)

        data = self.get_list_data(request, **kwargs)

        l = data.get('list')
        formset = None
        if l and l.formset:
            formset = l.formset

        url = self.request.build_absolute_uri()
        if formset:
            # Normally calling validate on a formset.
            # will result in a db call for each pk in
            # the formset regardless if the form has
            # changed or not.
            # To try to reduce queries only do a full
            # validate on forms that changed.
            # TODO: Find a way to not have to do
            # a pk lookup for any since we already
            # have the instance we want
            for form in formset.forms:
                if not form.has_changed():
                    form.cleaned_data = {}
                    form._errors = {}

        if formset and formset.is_valid():
            changecount = 0
            with transaction.commit_on_success():
                for form in formset.forms:
                    if form.has_changed():
                        obj = form.save()
                        changecount += 1
                        self.log_action(obj, CMSLog.SAVE, url=url,
                                        update_parent=changecount == 1)

            return self.render(request, redirect_url=url,
                           message="%s items updated" % changecount,
                           collect_render_data=False)
        else:
            return self.render(request, message = msg, **data)