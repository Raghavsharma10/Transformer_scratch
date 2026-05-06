def get_what_txt(self):
        """
        Overrides the base behaviour defined in ValidationError in order to add details about the class field.
        :return:
        """
        return 'field [{field}] for class [{clazz}]'.format(field=self.get_variable_str(),
                                                            clazz=self.validator.get_validated_class_display_name())