def transaction(self, transaction_id, expand_merchant=False):
        """
        Returns an individual transaction, fetched by its id.

        Official docs:
            https://monzo.com/docs/#retrieve-transaction

        :param transaction_id: Monzo transaction ID
        :type transaction_id: str
        :param expand_merchant: whether merchant data should be included
        :type expand_merchant: bool
        :returns: Monzo transaction details
        :rtype: MonzoTransaction
        """
        endpoint = '/transactions/{}'.format(transaction_id)

        data = dict()
        if expand_merchant:
            data['expand[]'] = 'merchant'

        response = self._get_response(
            method='get', endpoint=endpoint, params=data,
        )

        return MonzoTransaction(data=response.json()['transaction'])