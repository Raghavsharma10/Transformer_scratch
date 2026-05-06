def _validate_measure_count(self, times):
        """
        check if "times" is within the borders defined in the class

        :param times: "times" to check
        :type times: int
        """
        if not self.min_measures <= times <= self.max_measures:
            raise ParameterValidationError(
                "{times} is not within the borders defined in the class".format(
                    times=times
                )
            )