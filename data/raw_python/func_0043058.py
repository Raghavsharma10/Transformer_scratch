def post(self, request, *args, **kwargs):
        """
        Method for handling POST requests. Publishes
        the object passing the value of 'when' to the object's
        publish method. The object's `purge_archives` method
        is also called to limit the number of old items
        that we keep around. The action is logged as either
        'published' or 'scheduled' depending on the value of
        'when', and the user is notified with a message.

        Returns a 'render redirect' to the result of the
        `get_done_url` method.
        """

        self.object = self.get_object()
        form = self.form()
        url = self.get_done_url()
        if request.POST.get('publish'):
            form = self.form(request.POST)
            if form.is_valid():
                when = form.cleaned_data.get('when')
                self.object.publish(user=request.user, when=when)
                self.object.purge_archives()
                object_url = self.get_object_url()
                if self.object.state == self.object.PUBLISHED:
                    self.log_action(
                        self.object, CMSLog.PUBLISH, url=object_url)
                else:
                    self.log_action(
                        self.object, CMSLog.SCHEDULE, url=object_url)

                message = "%s %s" % (self.object, self.object.state)
                self.write_message(message=message)

                return self.render(request, redirect_url=url,
                           message=message,
                           obj=self.object,
                           collect_render_data=False)
        return self.render(request, obj=self.object, form=form, done_url=url)