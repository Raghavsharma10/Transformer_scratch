def as_serializable(self, attribute_filter=None, minified=False):
        """
        Returns a dictionary with attributes and pure python representation of
        the data instances. If an attribute filter is provided as_serializable
        will respect the visibility.

        The response is used by serializers to return data to client

        :param attribute_filter:
        :type attribute_filter: prestans.parser.AttributeFilter
        :param minified:
        :type minified: bool
        """
        from prestans.parser import AttributeFilter
        from prestans.parser import AttributeFilterImmutable
        from prestans.types import Array

        model_dictionary = dict()

        rewrite_map = self.attribute_rewrite_map()

        # convert filter to immutable if it isn't already
        if isinstance(attribute_filter, AttributeFilter):
            attribute_filter = attribute_filter.as_immutable()

        for attribute_name, type_instance in self.getmembers():

            if isinstance(attribute_filter, (AttributeFilter, AttributeFilterImmutable)) and \
               not attribute_filter.is_attribute_visible(attribute_name):
                continue

            # support minification
            serialized_attribute_name = attribute_name
            if minified is True:
                serialized_attribute_name = rewrite_map[attribute_name]

            if attribute_name not in self._attributes or self._attributes[attribute_name] is None:
                if isinstance(type_instance, Array):
                    model_dictionary[serialized_attribute_name] = []
                else:
                    model_dictionary[serialized_attribute_name] = None
                continue

            if isinstance(type_instance, DataCollection):

                sub_attribute_filter = None
                if isinstance(attribute_filter, (AttributeFilter, AttributeFilterImmutable)) and attribute_name in attribute_filter:
                    sub_attribute_filter = getattr(attribute_filter, attribute_name)

                model_dictionary[serialized_attribute_name] = self._attributes[attribute_name].as_serializable(sub_attribute_filter, minified)

            elif isinstance(type_instance, DataStructure):
                python_value = self._attributes[attribute_name]
                serializable_value = type_instance.as_serializable(python_value)
                model_dictionary[serialized_attribute_name] = serializable_value

            elif isinstance(type_instance, DataType):
                model_dictionary[serialized_attribute_name] = self._attributes[attribute_name]

        return model_dictionary