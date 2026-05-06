def get_fields_dict(self):
        """
        Return a list of `label`/`value` dictionaries to represent the
        fiels named by the model_admin class's `get_inspect_view_fields` method
        """
        fields = []
        for field_name in self.model_admin.get_inspect_view_fields():
            fields.append(self.get_dict_for_field(field_name))
        return fields