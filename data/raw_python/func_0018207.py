def collect(self, order_ref):
        """Collects the result of a sign or auth order using the
        ``orderRef`` as reference.

        RP should keep on calling collect every two seconds as long as status
        indicates pending. RP must abort if status indicates failed. The user
        identity is returned when complete.

        Example collect results returned while authentication or signing is
        still pending:

        .. code-block:: json

            {
                "orderRef":"131daac9-16c6-4618-beb0-365768f37288",
                "status":"pending",
                "hintCode":"userSign"
            }

        Example collect result when authentication or signing has failed:

        .. code-block:: json

            {
                "orderRef":"131daac9-16c6-4618-beb0-365768f37288",
                "status":"failed",
                "hintCode":"userCancel"
            }

        Example collect result when authentication or signing is successful
        and completed:

        .. code-block:: json

            {
                "orderRef":"131daac9-16c6-4618-beb0-365768f37288",
                "status":"complete",
                "completionData": {
                    "user": {
                        "personalNumber":"190000000000",
                        "name":"Karl Karlsson",
                        "givenName":"Karl",
                        "surname":"Karlsson"
                    },
                    "device": {
                        "ipAddress":"192.168.0.1"
                    },
                    "cert": {
                        "notBefore":"1502983274000",
                        "notAfter":"1563549674000"
                    },
                    "signature":"<base64-encoded data>",
                    "ocspResponse":"<base64-encoded data>"
                }
            }

        See `BankID Relying Party Guidelines Version: 3.0 <https://www.bankid.com/assets/bankid/rp/bankid-relying-party-guidelines-v3.0.pdf>`_
        for more details about how to inform end user of the current status,
        whether it is pending, failed or completed.

        :param order_ref: The ``orderRef`` UUID returned from auth or sign.
        :type order_ref: str
        :return: The CollectResponse parsed to a dictionary.
        :rtype: dict
        :raises BankIDError: raises a subclass of this error
                             when error has been returned from server.

        """
        response = self.client.post(
            self._collect_endpoint, json={"orderRef": order_ref}
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise get_json_error_class(response)