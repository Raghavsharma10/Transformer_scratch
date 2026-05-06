def process_action(self, request, queryset):
        """
        Publishes the selected objects by passing the value of \
        'when' to the object's publish method. The object's \
        `purge_archives` method is also called to limit the number \
        of old items that we keep around. The action is logged as \
        either 'published' or 'scheduled' depending on the value of \
        'when', and the user is notified with a message.

        Returns a 'render redirect' to the result of the \
        `get_done_url` method.
        """
        form = self.form(request.POST)
        if form.is_valid():
            when = form.cleaned_data.get('when')
            count = 0
            for obj in queryset:
                count += 1
                obj.publish(user=request.user, when=when)
                obj.purge_archives()
                object_url = self.get_object_url(obj)
                if obj.state == obj.PUBLISHED:
                    self.log_action(
                        obj, CMSLog.PUBLISH, url=object_url)
                else:
                    self.log_action(
                       obj, CMSLog.SCHEDULE, url=object_url)
            message = "%s objects published." % count
            self.write_message(message=message)

            return self.render(request, redirect_url= self.get_done_url(),
                                message=message,
                                collect_render_data=False)
        return self.render(request, queryset=queryset, publish_form=form, action='Publish')