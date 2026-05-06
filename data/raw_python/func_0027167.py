def is_password(self, is_password):
        """ Setter for is_identifier """

        if is_password:
            self.is_forgetable = True

        self._is_password = is_password