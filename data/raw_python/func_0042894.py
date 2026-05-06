def valid_value(self, value):
        """
        Check if the provided value is a valid choice.
        """
        if isinstance(value, Constant):
            value = value.name
        text_value = force_text(value)
        for option_value, option_label, option_title in self.choices:
            if value == option_value or text_value == force_text(option_value):
                return True
        return False