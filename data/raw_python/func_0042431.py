def do_scheduled_update(self, action, **kwargs):
        """
        Do the actual update.

        action: if provided it will be looked up
        on the implementing class and called with
        **kwargs. If action is not provided each k/v pair
        in kwargs will be set on self and then self
        is saved.

        kwargs: any other you passed for this update
        passed along to whichever method performs
        the update.
        """

        action = getattr(self, action, None)
        if callable(action):
            return action(**kwargs)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.save()