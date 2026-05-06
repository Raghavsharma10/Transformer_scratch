def get_field_display_value(self, field_name, field=None):
        """ Return a display value for a field """

        """
        Firstly, check for a 'get_fieldname_display' property/method on
        the model, and return the value of that, if present.
        """
        val_funct = getattr(self.instance, 'get_%s_display' % field_name, None)
        if val_funct is not None:
            if callable(val_funct):
                return val_funct()
            return val_funct

        """
        Secondly, if we have a real field, we can try to display something
        more useful for it.
        """
        if field is not None:
            try:
                field_type = field.get_internal_type()
                if (
                    field_type == 'ForeignKey' and
                    field.related_model == get_image_model()
                ):
                    # The field is an image
                    return self.get_image_field_display(field_name, field)

                if (
                    field_type == 'ForeignKey' and
                    field.related_model == Document
                ):
                    # The field is a document
                    return self.get_document_field_display(field_name, field)

            except AttributeError:
                pass

        """
        Resort to getting the value of 'field_name' from the instance.
        """
        return getattr(self.instance, field_name,
                       self.model_admin.get_empty_value_display())