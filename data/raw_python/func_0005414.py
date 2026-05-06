def _none_value(self):
        """Get an appropriate "null" value for this field's type. This
        is used internally when setting the field to None.
        """
        if self.out_type == int:
            return 0
        elif self.out_type == float:
            return 0.0
        elif self.out_type == bool:
            return False
        elif self.out_type == six.text_type:
            return u''