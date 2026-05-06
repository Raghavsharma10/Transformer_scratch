def is_valid(self,
                 value  # type: Any
                 ):
        # type: (...) -> bool
        """
        Validates the provided value and returns a boolean indicating success or failure. Any Exception happening in
        the validation process will be silently caught.

        :param value: the value to validate
        :return: a boolean flag indicating success or failure
        """
        # noinspection PyBroadException
        try:
            # perform validation
            res = self.main_function(value)

            # return a boolean indicating if success or failure
            return result_is_success(res)

        except Exception:
            # caught exception means failure > return False
            return False