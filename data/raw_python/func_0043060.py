def post(self, request, *args, **kwargs):
        """
        Method for handling POST requests. Unpublishes the
        the object by calling the object's unpublish method.
        The action is logged, the user is notified with a
        message. Returns a 'render redirect' to the result of
        the `get_done_url` method.
        """

        self.object = self.get_object()
        url = self.get_done_url()
        if request.POST.get('unpublish'):
            self.object.unpublish()
            object_url = self.get_object_url()
            self.log_action(self.object, CMSLog.UNPUBLISH, url=object_url)
            msg = self.write_message(message="%s unpublished" % (self.object))
            return self.render(request, redirect_url=url,
                       message=msg,
                       obj=self.object,
                       collect_render_data=False)

        return self.render(request, obj=self.object, done_url=url)