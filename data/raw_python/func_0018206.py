def sign(
        self,
        end_user_ip,
        user_visible_data,
        personal_number=None,
        requirement=None,
        user_non_visible_data=None,
        **kwargs
    ):
        """Request an signing order. The :py:meth:`collect` method
        is used to query the status of the order.

        Note that personal number is not needed when signing is to be done
        on the same device, provided that the returned ``autoStartToken``
        is used to open the BankID Client.

        Example data returned:

        .. code-block:: json

            {
                "orderRef":"131daac9-16c6-4618-beb0-365768f37288",
                "autoStartToken":"7c40b5c9-fa74-49cf-b98c-bfe651f9a7c6"
            }

        :param end_user_ip: IP address of the user requesting
            the authentication.
        :type end_user_ip: str
        :param user_visible_data: The information that the end user
            is requested to sign.
        :type user_visible_data: str
        :param personal_number: The Swedish personal number in
            format YYYYMMDDXXXX.
        :type personal_number: str
        :param requirement: An optional dictionary stating how the signature
            must be created and verified. See BankID Relying Party Guidelines,
            section 13.5 for more details.
        :type requirement: dict
        :param user_non_visible_data: Optional information sent with request
            that the user never sees.
        :type user_non_visible_data: str
        :return: The order response.
        :rtype: dict
        :raises BankIDError: raises a subclass of this error
                     when error has been returned from server.

        """
        data = {"endUserIp": end_user_ip}
        if personal_number:
            data["personalNumber"] = personal_number
        data["userVisibleData"] = self._encode_user_data(user_visible_data)
        if user_non_visible_data:
            data["userNonVisibleData"] = self._encode_user_data(user_non_visible_data)
        if requirement and isinstance(requirement, dict):
            data["requirement"] = requirement
        # Handling potentially changed optional in-parameters.
        data.update(kwargs)
        response = self.client.post(self._sign_endpoint, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise get_json_error_class(response)