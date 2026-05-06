def conforms_to_template_filter(self, template_filter):
        """
        Check AttributeFilter conforms to the rules set by the template

         - If self, has attributes that template_filter does not contain, throw Exception
         - If sub list found, perform the first check
         - If self has a value for an attribute, assign to final AttributeFilter
         - If not found, assign value from template

         todo: rename as current name is mis-leading
        """

        if not isinstance(template_filter, self.__class__):
            raise TypeError("AttributeFilter can only check conformance against \
                another template filter, %s provided" % template_filter.__class__.__name__)

        # keys from the template
        template_filter_keys = template_filter.keys()
        # Keys from the object itself
        this_filter_keys = self.keys()

        # 1. Check to see if the client has provided unwanted keys
        unwanted_keys = set(this_filter_keys) - set(template_filter_keys)
        if len(unwanted_keys) > 0:
            raise exception.AttributeFilterDiffers(list(unwanted_keys))

        # 2. Make a attribute_filter that we send back
        evaluated_attribute_filter = AttributeFilter()

        # 3. Evaluate the differences between the two, with template_filter as the standard
        for template_key in template_filter_keys:

            if template_key in this_filter_keys:

                value = getattr(self, template_key)

                # if sub filter and boolean provided with of true, create default filter with value of true
                if isinstance(value, bool) and \
                   value is True and \
                   isinstance(getattr(template_filter, template_key), AttributeFilter):
                    setattr(evaluated_attribute_filter, template_key, getattr(template_filter, template_key))
                elif isinstance(value, bool):
                    setattr(evaluated_attribute_filter, template_key, value)
                elif isinstance(value, self.__class__):
                    # Attribute lists sort themselves out, to produce sub Attribute Filters
                    template_sub_list = getattr(template_filter, template_key)
                    this_sub_list = getattr(self, template_key)
                    setattr(
                        evaluated_attribute_filter, template_key,
                        this_sub_list.conforms_to_template_filter(template_sub_list)
                    )
            else:
                setattr(evaluated_attribute_filter, template_key, getattr(template_filter, template_key))

        return evaluated_attribute_filter