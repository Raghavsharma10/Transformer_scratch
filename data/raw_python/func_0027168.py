def choices(self, choices):
        """ Setter for is_identifier """

        if choices is not None and len(choices) > 0:
            self.has_choices = True

        self._choices = choices