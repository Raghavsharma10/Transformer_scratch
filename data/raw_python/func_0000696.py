def key(self, *args, **kwargs):
        """
        Return the cache key to use.

        If you're passing anything but primitive types to the ``get`` method,
        it's likely that you'll need to override this method.
        """
        if not args and not kwargs:
            return self.class_path
        try:
            if args and not kwargs:
                return "%s:%s" % (self.class_path, self.hash(args))
            # The line might break if your passed values are un-hashable.  If
            # it does, you need to override this method and implement your own
            # key algorithm.
            return "%s:%s:%s:%s" % (self.class_path,
                                    self.hash(args),
                                    self.hash([k for k in sorted(kwargs)]),
                                    self.hash([kwargs[k] for k in sorted(kwargs)]))
        except TypeError:
            raise RuntimeError(
                "Unable to generate cache key due to unhashable"
                "args or kwargs - you need to implement your own"
                "key generation method to avoid this problem")