def set(self, option, value=None):
        """Set an option for chaining.

        Args:
            option (str): option name
            value (str): value, default None
        """
        option = self._container.optionxform(option)
        if option in self.options():
            self.__getitem__(option).value = value
        else:
            self.__setitem__(option, value)
        return self