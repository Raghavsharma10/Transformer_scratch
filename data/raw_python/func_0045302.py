def get_choices_for(self, field):
        """
        Get the choices for the given fields.

        Args:
            field (str): Name of field.

        Returns:
            List of tuples. [(name, value),...]
        """
        choices = self._fields[field].choices
        if isinstance(choices, six.string_types):
            return [(d['value'], d['name']) for d in self._choices_manager.get_all(choices)]
        else:
            return choices