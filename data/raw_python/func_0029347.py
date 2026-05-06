def check_aggregations_privacy(self, aggregations_params):
        """ Check per-field privacy rules in aggregations.

        Privacy is checked by making sure user has access to the fields
        used in aggregations.
        """
        fields = self.get_aggregations_fields(aggregations_params)
        fields_dict = dictset.fromkeys(fields)
        fields_dict['_type'] = self.view.Model.__name__

        try:
            validate_data_privacy(self.view.request, fields_dict)
        except wrappers.ValidationError as ex:
            raise JHTTPForbidden(
                'Not enough permissions to aggregate on '
                'fields: {}'.format(ex))