def _get_function_name(self):
        """
        Get function name of calling method

        :return: The name of the calling function
            (expected to be called in self.error/debug/..)
        :rtype: str | unicode
        """
        fname = inspect.getframeinfo(inspect.stack()[2][0]).function
        if fname == "<module>":
            return ""
        else:
            return fname