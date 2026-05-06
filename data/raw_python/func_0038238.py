def validate(self, value, attribute_filter=None, minified=False):
        """
        :param value: serializable input to validate
        :type value: dict | None
        :param attribute_filter:
        :type: prestans.parser.AttributeFilter | None
        :param minified: whether or not the input is minified
        :type minified: bool
        :return: the validated model
        :rtype: Model
        """

        if self._required and (value is None or not isinstance(value, dict)):
            """
            Model level validation requires a parsed dictionary
            this is done by the serializer
            """
            raise exception.RequiredAttributeError()

        if not self._required and not value:
            """
            Value was not provided by caller, but require a template
            """
            return None

        _model_instance = self.__class__()

        rewrite_map = self.attribute_rewrite_map()

        from prestans.parser import AttributeFilter
        from prestans.parser import AttributeFilterImmutable

        for attribute_name, type_instance in self.getmembers():
            if not isinstance(type_instance, DataType):
                raise TypeError("%s must be a DataType subclass" % attribute_name)

            if isinstance(attribute_filter, (AttributeFilter, AttributeFilterImmutable)) and \
               not attribute_filter.is_attribute_visible(attribute_name):
                _model_instance._attributes[attribute_name] = None
                continue

            validation_input = None

            input_value_key = attribute_name

            # minification support
            if minified is True:
                input_value_key = rewrite_map[attribute_name]

            if input_value_key in value:
                validation_input = value[input_value_key]

            try:

                if isinstance(type_instance, DataCollection):
                    sub_attribute_filter = None
                    if attribute_filter and attribute_name in attribute_filter:
                        sub_attribute_filter = getattr(attribute_filter, attribute_name)

                    validated_object = type_instance.validate(
                        validation_input,
                        sub_attribute_filter,
                        minified
                    )
                else:
                    validated_object = type_instance.validate(validation_input)

                _model_instance._attributes[attribute_name] = validated_object

            except exception.DataValidationException as exp:
                raise exception.ValidationError(
                    message=str(exp),
                    attribute_name=attribute_name,
                    value=validation_input,
                    blueprint=type_instance.blueprint()
                )

        return _model_instance