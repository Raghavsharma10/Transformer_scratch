def authenticate(self, personal_number, **kwargs):
        """Request an authentication order. The :py:meth:`collect` method
        is used to query the status of the order.

        :param personal_number: The Swedish personal number
            in format YYYYMMDDXXXX.
        :type personal_number: str
        :return: The OrderResponse parsed to a dictionary.
        :rtype: dict
        :raises BankIDError: raises a subclass of this error
                             when error has been returned from server.

        """
        if "requirementAlternatives" in kwargs:
            warnings.warn(
                "Requirement Alternatives " "option is not tested.", BankIDWarning
            )

        try:
            out = self.client.service.Authenticate(
                personalNumber=personal_number, **kwargs
            )
        except Error as e:
            raise get_error_class(e, "Could not complete Authenticate order.")

        return self._dictify(out)