def serialize(self, value):
        """Convert the external Python value to a type that is suitable for
        storing in a Mutagen file object.
        """
        if isinstance(value, float) and self.as_type is six.text_type:
            value = u'{0:.{1}f}'.format(value, self.float_places)
            value = self.as_type(value)
        elif self.as_type is six.text_type:
            if isinstance(value, bool):
                # Store bools as 1/0 instead of True/False.
                value = six.text_type(int(bool(value)))
            elif isinstance(value, bytes):
                value = value.decode('utf-8', 'ignore')
            else:
                value = six.text_type(value)
        else:
            value = self.as_type(value)

        if self.suffix:
            value += self.suffix

        return value