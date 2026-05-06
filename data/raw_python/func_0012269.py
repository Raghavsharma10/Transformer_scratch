def option(self, key, value=None, **kwargs):
        """Creates a new option inside a section

        Args:
            key (str): key of the option
            value (str or None): value of the option
            **kwargs: are passed to the constructor of :class:`Option`

        Returns:
            self for chaining
        """
        if not isinstance(self._container, Section):
            raise ValueError("Options can only be added inside a section!")
        option = Option(key, value, container=self._container, **kwargs)
        option.value = value
        self._container.structure.insert(self._idx, option)
        self._idx += 1
        return self