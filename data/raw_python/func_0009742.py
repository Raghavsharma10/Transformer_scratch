def get_app(self, reference_app=None):
        """Helper method that implements the logic to look up an application."""

        if reference_app is not None:
            return reference_app

        if self.app is not None:
            return self.app

        ctx = stack.top

        if ctx is not None:
            return ctx.app

        raise RuntimeError('Application not registered on Bouncer'
                           ' instance and no application bound'
                           ' to current context')