def selected(self, ids):
        """
        Return the selected options as a list of tuples
        """
        # Cleanup the ID list
        if self.get_field_name() == 'pk':
            ids = filter(lambda x: "{}".format(x).isdigit(), copy(ids))
        else:
            ids = filter(lambda x: len("{}".format(x)) > 0, copy(ids))
        # Prepare the QS
        # TODO: not contextually filtered, check if it's possible at some point
        qs = self.get_model_queryset().filter(
            **{'{}__in'.format(self.get_field_name()): ids})
        result = []
        for item in qs:
            item_repr = self.item(item)
            result.append(
                (item_repr['value'], item_repr['label'])
            )
        return result