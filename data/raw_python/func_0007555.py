def __set_transaction_detail(self, *args, **kwargs):
        """
        Checks kwargs for 'customer_transaction_id' and sets it if present.
        """

        customer_transaction_id = kwargs.get('customer_transaction_id', None)
        if customer_transaction_id:
            transaction_detail = self.client.factory.create('TransactionDetail')
            transaction_detail.CustomerTransactionId = customer_transaction_id
            self.logger.debug(transaction_detail)
            self.TransactionDetail = transaction_detail