def handleError(self, now, failureObj):
        """ An error occurred running my runnable.  Check my runnable for an
        error-handling method called 'timedEventErrorHandler' that will take
        the given failure as an argument, and execute that if available:
        otherwise, create a TimedEventFailureLog with information about what
        happened to this event.

        Must be run in a transaction.
        """
        errorHandler = getattr(self.runnable, 'timedEventErrorHandler', None)
        if errorHandler is not None:
            self._rescheduleFromRun(errorHandler(self, failureObj))
        else:
            self._defaultErrorHandler(now, failureObj)