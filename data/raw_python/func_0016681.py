def get_context_data(self, **kwargs):
        """
        Adds `next` to the context.

        This makes sure that the `next` parameter doesn't get lost if the
        form was submitted invalid.

        """
        ctx = super(UserMediaImageViewMixin, self).get_context_data(**kwargs)
        ctx.update({
            'action': self.action,
            'next': self.next,
        })
        return ctx