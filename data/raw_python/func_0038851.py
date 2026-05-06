def get_field_label(self, trans, field):
        """
        Get the field label from the _meta api of the model

        :param trans:
        :param field:
        :return:
        """
        try:
            # get from the instance
            object_field_label = trans._meta.get_field_by_name(field)[0].verbose_name
        except Exception:
            try:
                # get from the class
                object_field_label = self.sender._meta.get_field_by_name(field)[0].verbose_name
            except Exception:
                # in the worst case we set the field name as field label
                object_field_label = field
        return object_field_label