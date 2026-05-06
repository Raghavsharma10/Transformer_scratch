def wait(self, timeout=None):
        """
        Return the result, or throw the exception if result is a failure.

        It may take an unknown amount of time to return the result, so a
        timeout option is provided. If the given number of seconds pass with
        no result, a TimeoutError will be thrown.

        If a previous call timed out, additional calls to this function will
        still wait for a result and return it if available. If a result was
        returned or raised on one call, additional calls will return/raise the
        same result.
        """
        if threadable.isInIOThread():
            raise RuntimeError(
                "EventualResult.wait() must not be run in the reactor thread.")

        if imp.lock_held():
            try:
                imp.release_lock()
            except RuntimeError:
                # The lock is held by some other thread. We should be safe
                # to continue.
                pass
            else:
                # If EventualResult.wait() is run during module import, if the
                # Twisted code that is being run also imports something the
                # result will be a deadlock. Even if that is not an issue it
                # would prevent importing in other threads until the call
                # returns.
                raise RuntimeError(
                    "EventualResult.wait() must not be run at module "
                    "import time.")

        result = self._result(timeout)
        if isinstance(result, Failure):
            result.raiseException()
        return result