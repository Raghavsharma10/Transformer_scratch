def run(self, *args, **kwargs):
        """Call all the registered handlers with the arguments passed.
        If this signal is a class member, call also the handlers registered
        at class-definition time. If an external publish function is
        supplied, call it with the provided arguments.

        :returns: an instance of `~.utils.MultipleResults`
        """
        if self.fvalidation is not None:
            try:
                if self.fvalidation(*args, **kwargs) is False:
                    raise ExecutionError("Validation returned ``False``")
            except Exception as e:
                if __debug__:
                    logger.exception("Validation failed")
                else:
                    logger.error("Validation failed")
                raise ExecutionError(
                    "The validation of the arguments specified to ``run()`` "
                    "has failed") from e
        try:
            if self.exec_wrapper is None:
                return self.exec_all_endpoints(*args, **kwargs)
            else:
                # if a exec wrapper is defined, defer notification to it,
                # a callback to execute the default notification process
                result = self.exec_wrapper(self.endpoints,
                                           self.exec_all_endpoints,
                                           *args, **kwargs)
                if inspect.isawaitable(result):
                    result = pull_result(result)
                return result
        except Exception as e:
            if __debug__:
                logger.exception("Error while executing handlers")
            else:
                logger.error("Error while executing handlers")
            raise ExecutionError("Error while executing handlers") from e