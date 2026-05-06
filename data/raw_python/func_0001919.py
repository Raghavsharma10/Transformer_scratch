def name(self):
        """
        Get the module name

        :return: Module name
        :rtype: str | unicode
        """
        res = type(self).__name__
        if self._id:
            res += ".{}".format(self._id)
        return res