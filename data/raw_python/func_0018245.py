def collect(self, order_ref):
        """Collect the progress status of the order with the specified
        order reference.

        :param order_ref: The UUID string specifying which order to
            collect status from.
        :type order_ref: str
        :return: The CollectResponse parsed to a dictionary.
        :rtype: dict
        :raises BankIDError: raises a subclass of this error
                             when error has been returned from server.

        """
        try:
            out = self.client.service.Collect(order_ref)
        except Error as e:
            raise get_error_class(e, "Could not complete Collect call.")

        return self._dictify(out)