def item(self, current_item):
        """
        Return the current item.

        @param current_item: Current item
        @type  param: django.models

        @return: Value and label of the current item
        @rtype : dict
        """
        return {
            'value': text(getattr(current_item, self.get_field_name())),
            'label': self.label(current_item)
        }