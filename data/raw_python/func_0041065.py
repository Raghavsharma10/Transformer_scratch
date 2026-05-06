def reraise_if_any(failures, cause_cls_finder=None):
        """Re-raise exceptions if argument is not empty.

        If argument is empty list/tuple/iterator, this method returns
        None. If argument is converted into a list with a
        single ``Failure`` object in it, that failure is reraised. Else, a
        :class:`~.WrappedFailure` exception is raised with the failure
        list as causes.
        """
        if not isinstance(failures, (list, tuple)):
            # Convert generators/other into a list...
            failures = list(failures)
        if len(failures) == 1:
            failures[0].reraise(cause_cls_finder=cause_cls_finder)
        elif len(failures) > 1:
            raise WrappedFailure(failures)