def _warn_bogus_options(self, **opts):
        """
        Shows a warning for unsupported options for the current implementation.
        Called form set_options with remainig unsupported options.
        """
        if opts:
            import warnings
            for i in opts:
                warnings.warn("Unsupported option %s for %s" % (i, self), stacklevel=2)