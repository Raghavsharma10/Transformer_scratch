def reraise(self, cause_cls_finder=None):
        """Re-raise captured exception (possibly trying to recreate)."""
        if self._exc_info:
            six.reraise(*self._exc_info)
        else:
            # Attempt to regenerate the full chain (and then raise
            # from the root); without a traceback, oh well...
            root = None
            parent = None
            for cause in itertools.chain([self], self.iter_causes()):
                if cause_cls_finder is not None:
                    cause_cls = cause_cls_finder(cause)
                else:
                    cause_cls = None
                if cause_cls is None:
                    # Unable to find where this cause came from, give up...
                    raise WrappedFailure([self])
                exc = cause_cls(
                    *cause.exception_args, **cause.exception_kwargs)
                # Saving this will ensure that if this same exception
                # is serialized again that we will extract the traceback
                # from it directly (thus proxying along the original
                # traceback as much as we can).
                exc.__traceback_str__ = cause.traceback_str
                if root is None:
                    root = exc
                if parent is not None:
                    parent.__cause__ = exc
                parent = exc
            six.reraise(type(root), root, tb=None)