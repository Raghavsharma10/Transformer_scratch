def _response_item_to_object(self, resp_item):
        """
        take json and make a resource out of it
        """
        item_cls = resources.get_model_class(self.resource_type)
        properties_dict = resp_item[self.resource_type]
        new_dict = helpers.remove_properties_containing_None(properties_dict)
        # raises exception if something goes wrong
        obj = item_cls(new_dict)
        return obj