def authenticate(
        self, end_user_ip, personal_number=None, requirement=None, **kwargs
    ):
        """Request an authentication order. The :py:meth:`collect` method
        is used to query the status of the order.

        Note that personal number is not needed when authentication is to
        be done on the same device, provided that the returned
        ``autoStartToken`` is used to open the BankID Client.

        Example data returned:

        .. code-block:: json

            {
                "orderRef":"131daac9-16c6-4618-beb0-365768f37288",
                "autoStartToken":"7c40b5c9-fa74-49cf-b98c-bfe651f9a7c6"
            }

        :param end_user_ip: IP address of the user requesting
            the authentication.
        :type end_user_ip: str
        :param personal_number: The Swedish personal number in
            format YYYYMMDDXXXX.
        :type personal_number: str
        :param requirement: An optional dictionary stating how the signature
            must be created and verified. See BankID Relying Party Guidelines,
            section 13.5 for more details.
        :type requirement: dict
        :return: The order response.
        :rtype: dict
        :raises BankIDError: raises a subclass of this error
                             when error has been returned from server.

        """
        data = {"endUserIp": end_user_ip}
        if personal_number:
            data["personalNumber"] = personal_number
        if requirement and isinstance(requirement, dict):
            data["requirement"] = requirement
        # Handling potentially changed optional in-parameters.
        data.update(kwargs)
        response = self.client.post(self._auth_endpoint, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise get_json_error_class(response)