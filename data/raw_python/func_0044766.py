def format_label(self, field, counter):
        """
        Format the label for each field
        """
        return '<label for="id_formfield_%s" %s>%s</label>' % (
            counter, field.field.required and 'class="required"', field.label)