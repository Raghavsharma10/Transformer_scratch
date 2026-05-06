def _init_from_dictionary(self, from_dictionary, template_model=None):
        """
        Private helper to init values from a dictionary, wraps children into
        AttributeFilter objects

        :param from_dictionary: dictionary to get attribute names and visibility from
        :type from_dictionary: dict
        :param template_model:
        :type template_model: DataCollection
        """

        if not isinstance(from_dictionary, dict):
            raise TypeError("from_dictionary must be of type dict, %s \
                provided" % from_dictionary.__class__.__name__)

        rewrite_map = None
        if template_model is not None:

            if not isinstance(template_model, DataCollection):
                msg = "template_model should be a prestans model %s provided" % template_model.__class__.__name__
                raise TypeError(msg)

            rewrite_map = template_model.attribute_rewrite_reverse_map()

        for key, value in iter(from_dictionary.items()):

            target_key = key

            # minify support
            if rewrite_map is not None:
                target_key = rewrite_map[key]

            # ensure that the key exists in the template model
            if template_model is not None and target_key not in template_model:

                unwanted_keys = list()
                unwanted_keys.append(target_key)
                raise exception.AttributeFilterDiffers(unwanted_keys)

            # check to see we can work with the value
            if not isinstance(value, (bool, dict)):
                raise TypeError("AttributeFilter input for key %s must be \
                    boolean or dict, %s provided" % (key, value.__class__.__name__))

            # Either keep the value of wrap it up with AttributeFilter
            if isinstance(value, bool):
                setattr(self, target_key, value)
            elif isinstance(value, dict):

                sub_map = None
                if template_model is not None:

                    sub_map = getattr(template_model, target_key)

                    # prestans Array support
                    if isinstance(sub_map, Array):
                        sub_map = sub_map.element_template

                setattr(self, target_key, AttributeFilter(from_dictionary=value, template_model=sub_map))