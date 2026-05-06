def is_scalar(self):
        """
        :return:
        :rtype: bool
        """
        return \
            isinstance(self._element_template, Boolean) or \
            isinstance(self._element_template, Float) or  \
            isinstance(self._element_template, Integer) or \
            isinstance(self._element_template, String)