def process_action(self, request, queryset):
        """
        Unpublishes the selected objects by calling the object's \
        unpublish method. The action is logged and the user is \
        notified with a message.

        Returns a 'render redirect' to the result of the \
        `get_done_url` method.
        """
        count = 0
        for obj in queryset:
            count += 1
            obj.unpublish()
            object_url = self.get_object_url(obj)
            self.log_action(obj, CMSLog.UNPUBLISH, url=object_url)
        url = self.get_done_url()
        msg = self.write_message(message="%s objects unpublished." % count)
        return self.render(request, redirect_url=url,
                                message=msg,
                                collect_render_data=False)