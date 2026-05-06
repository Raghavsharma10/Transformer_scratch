def preview(self, when=timezone.now(), **kwargs):
        """Preview transactions, but don't actually save changes to list."""

        return self.operate_on(when=when, apply=False, **kwargs)