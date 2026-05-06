def run(self, func, *args, **kwargs):
        """Same as ``self.dryRun`` if ``self.dry``, else same as ``self.wetRun``."""
        if self.dry:
            return self.dryRun(func, *args, **kwargs)
        else:
            return self.wetRun(func, *args, **kwargs)