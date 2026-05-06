def safe_trigger(self, *args):
        """*Safely* triggers the event by invoking all its
        handlers, even if few of them raise an exception.

        If a set of exceptions is raised during handler
        invocation sequence, this method rethrows the first one.

        :param args: the arguments to invoke event handlers with.

        """
        error = None
        # iterate over a copy of the original list because some event handlers
        # may mutate the list
        for handler in list(self.handlers):
            try:
                handler(*args)
            except BaseException as e:
                if error is None:
                    prepare_for_reraise(e)
                    error = e
        if error is not None:
            reraise(error)