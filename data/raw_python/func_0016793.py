def transaction_read(self, items):
        """
        Wraps :func:`boto3.DynamoDB.Client.db.transact_get_items`.

        :param items: Unpacked into "TransactionItems" for :func:`boto3.DynamoDB.Client.transact_get_items`
        :raises bloop.exceptions.TransactionCanceled: if the transaction was canceled.
        :return: Dict with "Records" list
        """
        try:
            return self.dynamodb_client.transact_get_items(TransactItems=items)
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "TransactionCanceledException":
                raise TransactionCanceled from error
            raise BloopException("Unexpected error during transaction read.") from error