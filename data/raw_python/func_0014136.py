def raises(self, exception_type, function, *args, **kwargs):
        """
            Check if a function raises a specified exception type,
            *args and **kwargs are forwarded to the function
        """
        try:
            result = function(*args, **kwargs)
            self.log_error("{} did not throw exception {}".format(
                function.__name__,
                exception_type.__name__
            ), None)
            return result
        except Exception as e:
            if type(e) != exception_type:
                self.log_error("{} did raise {}: {}".format(
                    function.__name__,
                    type(e).__name__, e
                ), None)