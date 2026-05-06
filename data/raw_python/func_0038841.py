def get_obj_values(obj, translated_field_names):
        """
        get the translated field values from translatable fields of an object

        :param obj:
        :param translated_field_names:
        :return:
        """
        # set of translated fields to list
        fields = list(translated_field_names)
        values = {field: getattr(obj, field) for field in fields}
        return values