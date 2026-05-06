def validate(self, value, attribute_filter=None, minified=False):
        """
        :param value:
        :type value: list | None
        :param attribute_filter:
        :type attribute_filter: prestans.parser.AttributeFilter
        :param minified:
        :type minified: bool
        :return:
        """

        if not self._required and value is None:
            return None
        elif self._required and value is None:
            raise exception.RequiredAttributeError()

        _validated_value = self.__class__(
            element_template=self._element_template,
            min_length=self._min_length,
            max_length=self._max_length
        )

        if not isinstance(value, (list, tuple)):
            raise TypeError(value)

        for array_element in value:

            if isinstance(self._element_template, DataCollection):
                validated_array_element = self._element_template.validate(array_element, attribute_filter, minified)
            else:
                validated_array_element = self._element_template.validate(array_element)

            _validated_value.append(validated_array_element)

        if self._min_length is not None and len(_validated_value) < self._min_length:
            raise exception.LessThanMinimumError(value, self._min_length)

        if self._max_length is not None and len(_validated_value) > self._max_length:
            raise exception.MoreThanMaximumError(value, self._max_length)

        return _validated_value