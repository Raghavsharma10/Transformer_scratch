def set_field_value(self, field_name, value):
        """ Set value of request field named `field_name`.

        Use this method to apply changes to object which is affected
        by request. Values are set on `view._json_params` dict.

        If `field_name` is not affected by request, it is added to
        `self.fields` which makes field processors which are connected
        to `field_name` to be triggered, if they are run after this
        method call(connected to events after handler that performs
        method call).

        :param field_name: Name of request field value of which should
            be set.
        :param value: Value to be set.
        """
        self.view._json_params[field_name] = value
        if field_name in self.fields:
            self.fields[field_name].new_value = value
            return

        fields = FieldData.from_dict({field_name: value}, self.model)
        self.fields.update(fields)