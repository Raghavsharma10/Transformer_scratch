def get_response_attribute_filter(self, template_filter, template_model=None):
        """
        Prestans-Response-Attribute-List can contain a client's requested
        definition for attributes required in the response. This should match
        the response_attribute_filter_template?

        :param template_filter:
        :param template_model: the expected model that this filter corresponds to
        :return:
        :rtype: None | AttributeFilter
        """

        if template_filter is None:
            return None

        if 'Prestans-Response-Attribute-List' not in self.headers:
            return None

        # header not set results in a None
        attribute_list_str = self.headers['Prestans-Response-Attribute-List']

        # deserialize the header contents
        json_deserializer = deserializer.JSON()
        attribute_list_dictionary = json_deserializer.loads(attribute_list_str)

        # construct an AttributeFilter
        attribute_filter = AttributeFilter(
            from_dictionary=attribute_list_dictionary,
            template_model=template_model
        )

        #: Check template? Do this even through we might have template_model
        #: in case users have made a custom filter
        evaluated_filter = attribute_filter.conforms_to_template_filter(template_filter)

        return evaluated_filter