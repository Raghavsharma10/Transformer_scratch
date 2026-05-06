def _validate_json_format(self, filter_value, schema_validation_type):
        """
        adds the type:string format:schema_validation_type
        :param filter_value: value of the filter
        :param schema_validation_type: format description of the json schema entry
        """
        ok = False
        try:
            validators.json_schema_validation_format(filter_value, schema_validation_type)
            ok = True
        except ValueError as e:
            pass
        return ok