def schedule(self, when=None, action=None, **kwargs):
        """
        Schedule an update of this object.

        when: The date for the update.

        action: if provided it will be looked up
        on the implementing class and called with
        **kwargs. If action is not provided each k/v pair
        in kwargs will be set on self and then self
        is saved.

        kwargs: any other arguments you would like passed
        for this change. Saved as a json object so must cleanly
        serialize.
        """

        # when is empty or passed, just save it now.
        if not when or when <= timezone.now():
            self.do_scheduled_update(action, **kwargs)
        else:
            ctype = ContentType.objects.get_for_model(self.__class__)
            Schedule(
                content_type=ctype,
                object_args=self.get_scheduled_filter_args(),
                when=when,
                action=action,
                json_args=kwargs
            ).save()