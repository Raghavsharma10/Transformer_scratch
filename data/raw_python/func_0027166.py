def is_identifier(self, is_identifier):
        """ Setter for is_identifier """

        if is_identifier:
            self.is_editable = False

        self._is_identifier = is_identifier