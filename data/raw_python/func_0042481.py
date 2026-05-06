def schedule(self, when=None, action=None, **kwargs):
        """
        Schedule this item to be published.

        :param when: Date/time when this item should go live. None means now.
        """
        action = '_publish'
        super(BaseVersionedModel, self).schedule(when=when, action=action,
                                                 **kwargs)