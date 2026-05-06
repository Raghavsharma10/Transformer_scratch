def sign(self, user_visible_data, personal_number=None, **kwargs):
        """Request an signing order. The :py:meth:`collect` method
        is used to query the status of the order.

        :param user_visible_data: The information that the end user is
            requested to sign.
        :type user_visible_data: str
        :param personal_number: The Swedish personal number in
            format YYYYMMDDXXXX.
        :type personal_number: str
        :return: The OrderResponse parsed to a dictionary.
        :rtype: dict
        :raises BankIDError: raises a subclass of this error
                     when error has been returned from server.

        """
        if "requirementAlternatives" in kwargs:
            warnings.warn(
                "Requirement Alternatives option is not tested.", BankIDWarning
            )

        if isinstance(user_visible_data, six.text_type):
            data = base64.b64encode(user_visible_data.encode("utf-8")).decode("ascii")
        else:
            data = base64.b64encode(user_visible_data).decode("ascii")

        try:
            out = self.client.service.Sign(
                userVisibleData=data, personalNumber=personal_number, **kwargs
            )
        except Error as e:
            raise get_error_class(e, "Could not complete Sign order.")

        return self._dictify(out)