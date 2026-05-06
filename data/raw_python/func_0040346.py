def close(self):
        """Close injector & injected Provider instances, including generators.

        Providers are closed in the reverse order in which they were opened,
        and each provider is only closed once. Providers are closed if accessed
        by the injector, even if a dependency is not successfully provided. As
        such, providers should determine whether or not anything needs to be
        done in the close method.
        """
        if self.closed:
            raise RuntimeError('{!r} already closed'.format(self))
        for finalizer in reversed(self.finalizers):
            # Note: Unable to apply injector on close method.
            finalizer()
        self.closed = True
        self.instances.clear()
        self.values.clear()