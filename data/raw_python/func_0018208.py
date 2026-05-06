def cancel(self, order_ref):
        """Cancels an ongoing sign or auth order.

        This is typically used if the user cancels the order
        in your service or app.

        :param order_ref: The UUID string specifying which order to cancel.
        :type order_ref: str
        :return: Boolean regarding success of cancellation.
        :rtype: bool
        :raises BankIDError: raises a subclass of this error
                             when error has been returned from server.

        """
        response = self.client.post(self._cancel_endpoint, json={"orderRef": order_ref})

        if response.status_code == 200:
            return response.json() == {}
        else:
            raise get_json_error_class(response)